# Configuration Reference

All project constants are defined in `src/constant/constant.py`. This document provides a full reference for every configurable parameter with its default value, type, and rationale.

> **Rule:** Every downstream class and function imports exclusively from `constant.py`. Changing a value here propagates automatically throughout the entire pipeline — no other file needs editing.

---

## Paths

| Constant           | Default Value                                        | Description                                     |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------- |
| `BASE_DIR`         | `Path(__file__).resolve().parent.parent.parent`      | Project root directory                          |
| `DATASET_ROOT`     | `BASE_DIR / "data"`                                  | Local dataset directory                         |
| `CHECKPOINT_PATH`  | `BASE_DIR / "checkpoint" / "spiral_best_model.pth"`  | Best PyTorch checkpoint (saved during training) |
| `ONNX_EXPORT_PATH` | `BASE_DIR / "checkpoint" / "spiral_best_model.onnx"` | ONNX export for production inference            |

> When running locally, update `DATASET_ROOT` to point to your Parkinson's drawings dataset location.

---

## Image

| Constant     | Type              | Default      | Description                                     |
| ------------ | ----------------- | ------------ | ----------------------------------------------- |
| `IMAGE_SIZE` | `Tuple[int, int]` | `(224, 224)` | Target `(width, height)` passed to `cv2.resize` |

EfficientNet-B0 expects `224×224` input. Changing this requires recomputing HOG feature dimensions and retraining from scratch.

---

## Data Splitting

| Constant           | Type    | Default | Description                                        |
| ------------------ | ------- | ------- | -------------------------------------------------- |
| `TEST_SPLIT_RATIO` | `float` | `0.2`   | Fraction of data reserved as the held-out test set |
| `RANDOM_SEED`      | `int`   | `42`    | Seed for all reproducible splits                   |

The remaining 80% is further split internally during training (90% train / 10% val), giving an effective breakdown of **72% train / 8% val / 20% test**.

---

## HOG Feature Extraction

| Constant              | Type              | Default    | Description                             |
| --------------------- | ----------------- | ---------- | --------------------------------------- |
| `HOG_ORIENTATIONS`    | `int`             | `9`        | Number of gradient orientation bins     |
| `HOG_PIXELS_PER_CELL` | `Tuple[int, int]` | `(16, 16)` | Size of each HOG cell in pixels         |
| `HOG_CELLS_PER_BLOCK` | `Tuple[int, int]` | `(2, 2)`   | Number of cells per normalisation block |

### HOG Output Vector Length

```
cells_x   = 224 // 16 = 14
cells_y   = 224 // 16 = 14
blocks_x  = 14 - 2 + 1 = 13
blocks_y  = 14 - 2 + 1 = 13

HOG length = 13 × 13 × 2 × 2 × 9 = 6,084
```

---

## LBP Feature Extraction

| Constant         | Type              | Default                  | Description                                 |
| ---------------- | ----------------- | ------------------------ | ------------------------------------------- |
| `LBP_RADIUS`     | `int`             | `1`                      | Radius of the circular LBP neighbourhood    |
| `LBP_N_POINTS`   | `int`             | `8` (= 8 × `LBP_RADIUS`) | Number of sampling points around each pixel |
| `LBP_HIST_BINS`  | `int`             | `10`                     | Number of histogram bins for LBP patterns   |
| `LBP_HIST_RANGE` | `Tuple[int, int]` | `(0, 10)`                | Value range of the LBP histogram            |

The histogram is normalized to sum to 1 (with ε=1e-6 for numerical stability), producing a **10-dimensional** probability distribution vector.

---

## Combined Feature Vector

```
HOG features : 6,084
LBP features :    10
─────────────────────
Total         : 6,094
```

> **Note:** This raw 6,094-dim vector is further reduced by the Feature Reduction pipeline (VT → Scaler → PCA) before being fed to the model. The final dimension `K` passed to `MATH_FEATURE_DIM` is determined at runtime.

---

## Data Augmentation

| Constant                      | Type  | Default | Description                                            |
| ----------------------------- | ----- | ------- | ------------------------------------------------------ |
| `NUM_AUGMENTATIONS_PER_IMAGE` | `int` | `5`     | Augmented copies generated per original training image |

**Effective training set expansion:**

```
augmented_train = original_train × (1 + 5) = original_train × 6
```

Augmentation is applied **only to the training set** after the train/test split. The test set is never augmented.

---

## DataLoaders

| Constant           | Type  | Default | Description                                |
| ------------------ | ----- | ------- | ------------------------------------------ |
| `TRAIN_BATCH_SIZE` | `int` | `16`    | Training DataLoader batch size             |
| `VAL_BATCH_SIZE`   | `int` | `8`     | Validation DataLoader batch size           |
| `TEST_BATCH_SIZE`  | `int` | `4`     | Test DataLoader batch size (shuffle=False) |

---

## Optimiser

| Constant        | Type    | Default | Description                   |
| --------------- | ------- | ------- | ----------------------------- |
| `LEARNING_RATE` | `float` | `1e-4`  | Peak learning rate for AdamW  |
| `WEIGHT_DECAY`  | `float` | `1e-4`  | L2 regularisation coefficient |

AdamW is used rather than Adam because it decouples weight decay from the gradient update step, which is the mathematically correct formulation for L2 regularization with adaptive optimisers.

---

## LR Scheduler

| Constant               | Type        | Default | Description                                                  |
| ---------------------- | ----------- | ------- | ------------------------------------------------------------ |
| `WARMUP_START_FACTOR`  | `float`     | `0.01`  | LR multiplier at epoch 1 → `1e-4 × 0.01 = 1e-6`              |
| `WARMUP_END_FACTOR`    | `float`     | `1.0`   | LR multiplier at end of warmup → `1e-4`                      |
| `WARMUP_TOTAL_ITERS`   | `int`       | `5`     | Epochs for the linear warmup phase                           |
| `COSINE_T_MAX`         | `int`       | `10`    | Half-period of the cosine annealing cycle                    |
| `SCHEDULER_MILESTONES` | `List[int]` | `[5]`   | Epoch at which `SequentialLR` switches from warmup to cosine |

---

## Training

| Constant     | Type  | Default | Description           |
| ------------ | ----- | ------- | --------------------- |
| `NUM_EPOCHS` | `int` | `15`    | Total training epochs |

---

## Model Internals

| Constant          | Type  | Default             | Description                                                   |
| ----------------- | ----- | ------------------- | ------------------------------------------------------------- |
| `BACKBONE_NAME`   | `str` | `"efficientnet_b0"` | `timm` model identifier for the CNN backbone                  |
| `MATH_HIDDEN_DIM` | `int` | `256`               | Hidden dimension of the handcrafted feature MLP (first layer) |
| `MATH_OUTPUT_DIM` | `int` | `128`               | Output dimension of the handcrafted feature MLP               |

> `MATH_FEATURE_DIM` (the PCA-reduced input size) is **not a fixed constant** — it is determined at runtime from the training data and passed dynamically to the model constructor.

---

## Changing Defaults

To override any constant, edit `src/constant/constant.py` directly.

### Example — Changing the backbone

```python
# In constant.py
BACKBONE_NAME = "efficientnet_b2"
```

The `BaseModel` class reads `self.backbone.num_features` dynamically, so the fusion head input size adjusts automatically. No other code change is needed.

### Example — Increasing augmentation

```python
# In constant.py
NUM_AUGMENTATIONS_PER_IMAGE = 10
```

Be aware that higher augmentation multiplies training time proportionally and may increase overfitting risk on very small datasets.

### Example — Tighter feature reduction

```python
# In constant.py (PCA call)
n_components = 0.90   # Keep 90% variance instead of 95%
```

This produces a smaller `K`, faster training, but may lose discriminative information.
