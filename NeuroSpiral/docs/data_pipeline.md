# Data Pipeline

This document covers every stage of data handling: dataset structure, metadata extraction, image preprocessing, train/test splitting, augmentation, handcrafted feature extraction, feature reduction, and DataLoader construction.

---

## Dataset

**Source:** [Parkinson's Drawings — Kaggle (kmader)](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)

The dataset contains hand-drawn spiral and wave images from both healthy controls and Parkinson's disease patients, collected under clinical conditions.

### Directory Layout

```
parkinsons-drawings/
├── spiral/
│   ├── training/
│   │   ├── healthy/
│   │   └── patient/
│   └── testing/
│       ├── healthy/
│       └── patient/
└── wave/
    ├── training/
    │   ├── healthy/
    │   └── patient/
    └── testing/
        ├── healthy/
        └── patient/
```

### Class Labels

| Code | Meaning | Binary Value |
|------|---------|--------------|
| `HC` | Healthy Control | `1.0` |
| `PD` | Parkinson's Disease | `0.0` |

---

## Pipeline Overview

```
Dataset Directory
      │
      ▼
 MetadataBuilder
 (path, type_test, label)
      │
      ▼
 ImagePreprocessor
 Resize → LPF → Otsu → Morphology
      │
      ▼
 train_test_split  (80% / 20%)
      │
      ├───────────────────────────────┐
      │                               │
   Train (80%)                  Test (20%)
      │                          (held out — never augmented)
      ▼
 DataAugmentor
 ×6 expansion (original + 5 augmented)
      │
      ▼
 FeatureExtractor
 HOG (6,084) + LBP (10) = 6,094 features
      │
      ▼
 Feature Reduction
 VarianceThreshold → StandardScaler → PCA (95%)
      │
      ▼
 train_val_split (90% / 10%)
      │
      ├── Train DataLoader  (batch=16, shuffle=True)
      ├── Val DataLoader    (batch=8,  shuffle=True)
      └── Test DataLoader   (batch=4,  shuffle=False)
```

---

## Stage 1 — MetadataBuilder

**Class:** `MetaDataFactory`

Recursively scans the `spiral/` and `wave/` directories and yields one record per image.

### Output DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| `path` | `str` | Absolute path to the image file |
| `type_test` | `str` | `"spiral"` or `"wave"` |
| `label` | `str` | `"HC"` (healthy) or `"PD"` (patient) |

Label inference: folder name is lowercased and checked for the substring `"healthy"`.

---

## Stage 2 — ImagePreprocessor

**Class:** `ImagePreprocessor`

A **modular, configurable** preprocessing pipeline. Each step can be independently enabled or disabled via boolean flags.

### Processing Steps

```
cv2.imread → cv2.resize → [Fourier LPF] → [Otsu Binarization] → [Morphology]
```

---

#### Step 1 — Load & Resize

```python
cv2.imread(path)
cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
```

Output: BGR image at `(224, 224)`.

---

#### Step 2 — Fourier Low-Pass Filter *(optional, default: on)*

Removes high-frequency noise (tremor artifacts) in the frequency domain:

```python
cutoff = 30   # radius in frequency space
```

1. Convert BGR → grayscale
2. Compute 2D FFT and shift zero-frequency to centre
3. Apply circular mask of radius `cutoff`
4. Inverse FFT → normalize to [0, 255]

A **circular mask** is used (not square) for isotropic frequency cutoff. Output: grayscale `uint8`.

---

#### Step 3 — Otsu Binarization *(optional, default: on)*

```python
cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Automatic global thresholding — no manual threshold parameter required. Output: binary `uint8` image.

---

#### Step 4 — Morphological Cleaning *(optional, default: on)*

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN,  kernel, iterations=1)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
```

| Operation | Effect |
|-----------|--------|
| Opening | Removes small noise pixels |
| Closing | Fills small gaps in strokes |

---

### Output Structure

```python
{
    "img":       np.ndarray,   # shape (N, H, W), dtype uint8, single-channel
    "type_test": list[str],
    "label":     list[str],
}
```

Images are **grayscale** throughout the pipeline — this is intentional. Binary stroke images contain all the discriminative information needed, and grayscale avoids unnecessary colour channel noise.

### Error Handling

Corrupt or missing images are silently skipped and collected in an internal `skipped` list. Processing continues without interruption.

---

## Stage 3 — Train / Test Split

```python
train_test_split(data, test_size=0.2, stratify=data["label"], random_state=42)
```

| Split | Size | Notes |
|-------|------|-------|
| Train | 80% | Receives augmentation |
| Test | 20% | Held out — never touched until final evaluation |

Stratified splitting ensures both `HC` and `PD` classes are proportionally represented in both splits.

---

## Stage 4 — DataAugmentor

**Class:** `DataAugmentor` — applied **only to training data**.

### Augmentation Pipeline

```python
A.Compose([
    A.Rotate(limit=20, p=0.6),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.03, 0.03), shear=(-3, 3), p=0.5),
    A.GaussNoise(var_limit=(3.0, 10.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
])
```

| Transform | Purpose |
|-----------|---------|
| `Rotate(±20°)` | Simulates drawing angle variation |
| `Affine` | Simulates scale, translation, and shear variation |
| `GaussNoise` | Simulates tremor artifacts (most important for PD) |
| `RandomBrightnessContrast` | Simulates pen pressure / scanner variation |

### Expansion Factor

```
num_augmentations_per_image = 5
effective_multiplier = 1 (original) + 5 = ×6
```

Each original image is kept and 5 augmented copies are generated. The augmented dataset is then shuffled.

---

## Stage 5 — FeatureExtractor

**Class:** `FeatureExtractor` — extracts handcrafted descriptors from each (preprocessed) image.

### HOG — Histogram of Oriented Gradients

Captures **stroke direction and edge structure**:

```python
hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
```

Output: **6,084-dim** vector.

---

### LBP — Local Binary Pattern

Captures **local texture irregularities** (e.g., rough vs. smooth strokes):

```python
radius   = 1
n_points = 8   # = 8 × radius
method   = "uniform"

lbp = local_binary_pattern(img, n_points, radius, method="uniform")
lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
lbp_hist /= (lbp_hist.sum() + 1e-6)   # normalize
```

Output: **10-dim** normalized histogram vector.

---

### Combined Feature Vector

```
HOG : 6,084
LBP :    10
─────────────
Total: 6,094
```

---

## Stage 6 — Feature Reduction Pipeline

Three-step pipeline to reduce dimensionality, remove noise, and prevent data leakage.

### Critical Rule: Fit on Train, Transform on Val/Test

```python
# Training only — fit AND transform
X_pca, vt, scaler, pca = fit_reducers(train_data)

# Validation and test — transform only (no re-fitting)
X_pca = transform_only(data, vt, scaler, pca)
```

Fitting the reducers on validation or test data would constitute **data leakage**.

---

### Step 1 — VarianceThreshold

```python
VarianceThreshold(threshold=0.01)
```

Removes features with near-zero variance across the training set — these carry no discriminative information.

---

### Step 2 — StandardScaler

```python
StandardScaler()  # mean=0, std=1 per feature
```

Required before PCA to prevent high-magnitude features from dominating the principal components.

---

### Step 3 — PCA

```python
PCA(n_components=0.95, random_state=42)
```

Retains the minimum number of components that explain **95% of total variance**.

| Before | After |
|--------|-------|
| 6,094 features | ~100–400 features (data-dependent) |

Benefits: faster training, reduced overfitting, cleaner gradient signal.

---

## Stage 7 — Train / Val Split

```python
train_test_split(all_data, test_size=0.1, stratify=all_data["label"], random_state=42)
```

Split is performed **after augmentation and feature reduction** on the training portion only.

| Split | Size of training 80% |
|-------|----------------------|
| Train | 90% |
| Val | 10% |

---

## PyTorch Dataset — `SpiralData`

Each `__getitem__` call returns a tuple of three tensors:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `img_tensor` | `(3, H, W)` | Grayscale image replicated across 3 channels, normalised to [0, 1] |
| `math_features` | `(K,)` | PCA-reduced feature vector |
| `label` | `scalar` | `1.0` = HC, `0.0` = PD |

The 3-channel replication satisfies EfficientNet's input requirements while preserving the grayscale preprocessing.

---

## DataLoaders

| Split | Batch Size | Shuffle |
|-------|------------|---------|
| Train | 16 | Yes |
| Val | 8 | Yes |
| Test | 4 | No |

Test DataLoader has `shuffle=False` to ensure deterministic evaluation order.

---

## Summary of Design Decisions

| Decision | Rationale |
|----------|-----------|
| Fourier LPF before binarization | Removes tremor-induced high-frequency noise before thresholding |
| Otsu binarization | Automatic, parameter-free; robust to illumination changes |
| Morphological cleaning | Removes noise pixels and fills stroke gaps post-binarization |
| PCA fit only on train | Prevents data leakage from val/test statistics |
| Augmentation before feature extraction | Features are extracted from augmented images, maximising diversity |
| Stratified splits | Maintains class balance across all splits |