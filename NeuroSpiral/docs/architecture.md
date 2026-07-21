# Architecture

This document describes the design of **NeuroSpiral** — a hybrid neural network that fuses CNN image representations with handcrafted descriptors using an **adaptive gating mechanism**.

---

## Design Philosophy

Parkinson's Disease affects fine motor control, producing characteristic drawing impairments:

| Signal                     | Description                                       |
| -------------------------- | ------------------------------------------------- |
| Tremor noise               | High-frequency micro-oscillations in stroke paths |
| Irregular stroke thickness | Involuntary pressure variation                    |
| Inconsistent spacing       | Loss of smooth motor sequencing                   |

To capture these signals effectively, the model combines three complementary components:

| Component                 | Role                                                                           |
| ------------------------- | ------------------------------------------------------------------------------ |
| **EfficientNet-B0**       | Learns global spatial patterns from the full image                             |
| **HOG + LBP descriptors** | Capture local structure and texture irregularities                             |
| **Gating mechanism**      | Dynamically controls how much handcrafted signal should influence the decision |

> **Core Idea:** Not all samples need handcrafted features equally. The model learns **when to trust them** — suppressing them when noisy, amplifying them when informative.

---

## System Overview

```
Input Image (B, 3, H, W)          Math Features (B, K)
        │                                  │
        ▼                                  ▼
 ┌────────────────┐               ┌─────────────────┐
 │  EfficientNet  │               │   LayerNorm(K)  │
 │    Backbone    │               └────────┬────────┘
 │  (pretrained)  │                        │
 └───────┬────────┘               ┌────────▼────────┐
         │                        │  MLP Projection │
         │                        │  K → 256 → 128  │
         ▼                        └────────┬────────┘
   Image Features                          │
     (B, 1280)                    Math Embedding
         │                           (B, 128)
         │                                │
         └──────────┬─────────────────────┘
                    │
                    ▼
         Concatenation (B, 1408)
                    │
                    ▼
          ┌─────────────────┐
          │  Gate Network   │
          │  1408 → 128     │
          │  Sigmoid output │
          └────────┬────────┘
                   │
                   ▼
       Gate vector ∈ (0, 1)^128
                   │
       Applied to Math Embedding
       math_feat = math_feat × gate
                   │
                   ▼
    Final Concatenation (B, 1408)
    [Image Features + Gated Math]
                   │
                   ▼
           Fusion Head
        1408 → 256 → 64 → 1
                   │
                   ▼
                Logit
         (sigmoid → probability)
```

---

## Image Branch — EfficientNet-B0

| Property         | Value                                |
| ---------------- | ------------------------------------ |
| Model            | EfficientNet-B0 (via `timm`)         |
| Pretrained       | Yes (ImageNet-1K)                    |
| Pooling          | Global Average (`global_pool="avg"`) |
| Output dimension | 1280                                 |
| Fine-tuning      | Full — no frozen layers              |

The backbone receives a **3-channel tensor** replicated from the preprocessed grayscale image:

```python
img_tensor = img_tensor.unsqueeze(0).repeat(3, 1, 1)  # (1, H, W) → (3, H, W)
```

This preserves grayscale preprocessing benefits while satisfying the backbone's 3-channel input requirement.

---

## Handcrafted Feature Branch

### Input

Features enter the branch **after PCA reduction** (fitted on training data only):

```
Original HOG + LBP: 6143 dims
After VT + Scaler + PCA (95% variance): ~100–400 dims (K)
```

The exact value of `K` is determined at runtime and passed to the model as `math_feature_dim`.

---

### Normalization

```python
nn.LayerNorm(math_feature_dim)
```

Stabilizes the feature distribution **per sample**, independent of batch statistics — preferred over `BatchNorm` for non-image vectors.

---

### MLP Projection

```
Linear(K → 256)  →  BatchNorm1d  →  GELU  →  Dropout(0.3)
Linear(256 → 128) →  BatchNorm1d  →  GELU  →  Dropout(0.4)
```

Produces a compact 128-dim embedding from the handcrafted features.

---

## Gating Mechanism

### Motivation

Handcrafted features are not uniformly reliable across samples:

- **Clean drawings** → features carry useful structural signal
- **Noisy drawings** → features may capture tremor artifacts rather than class-discriminative patterns

The gate learns to **suppress or amplify** each dimension of the math embedding depending on the input image context.

---

### Gate Computation

```
Input: concat(img_feat [1280], math_feat [128]) = [1408]

Linear(1408 → 128) → GELU → Linear(128 → 128) → Sigmoid
```

Output: gate vector ∈ (0, 1)^128

---

### Applying the Gate

```python
math_feat = math_feat * gate   # element-wise scaling
```

This acts as **soft feature selection** — dimensions close to 0 are suppressed, dimensions close to 1 are passed through.

---

## Fusion Head

```
Input: concat(img_feat [1280], gated_math_feat [128]) = [1408]

Linear(1408 → 256) → GELU → Dropout(0.5)
Linear(256 → 64)   → GELU → Dropout(0.5)
Linear(64 → 1)     → Logit
```

The head is deliberately small and heavily regularized (Dropout 0.5) to prevent overfitting on the limited dataset size.

---

## Activation Choice — GELU

All non-output activations use GELU (Gaussian Error Linear Unit) rather than ReLU:

| Property      | ReLU                | GELU                           |
| ------------- | ------------------- | ------------------------------ |
| Smoothness    | Hard cutoff at 0    | Smooth probabilistic gate      |
| Gradient at 0 | Zero (dead neurons) | Non-zero                       |
| Performance   | Strong baseline     | Better in modern architectures |

GELU is used throughout Transformers, BERT, GPT, and EfficientNet variants — making it a consistent choice here.

---

## Loss Function

```python
nn.BCEWithLogitsLoss()
```

Combines sigmoid + binary cross-entropy in a single numerically stable operation using the log-sum-exp trick. Always preferred over `sigmoid` + `BCELoss`.

**Label smoothing** is applied during training:

```python
smooth = 0.1
label_smooth = label * (1 - smooth) + 0.5 * smooth
```

This prevents overconfident predictions and reduces overfitting on small datasets.

---

## Forward Pass (Annotated)

```python
def forward(self, img, math_features):
    # --- Image branch ---
    img_feat = self.backbone(img)                   # (B, 1280)

    # --- Math branch ---
    math_features = self.math_norm(math_features)   # LayerNorm
    math_feat = self.math_fc(math_features)         # (B, 128)

    # --- Gating ---
    combined = torch.cat([img_feat, math_feat], 1)  # (B, 1408)
    gate = self.gate_layer(combined)                # (B, 128) ∈ (0,1)
    math_feat = math_feat * gate                    # element-wise

    # --- Fusion ---
    combined = torch.cat([img_feat, math_feat], 1)  # (B, 1408)
    logit = self.fusion_layer(combined)             # (B, 1)

    return logit
```

---

## Parameter Count

| Component                | Parameters    |
| ------------------------ | ------------- |
| EfficientNet-B0 backbone | ~5.3M         |
| Math branch MLP          | ~0.1–0.3M     |
| Gate network             | ~0.2M         |
| Fusion head              | ~0.1M         |
| **Total**                | **~5.7–6.0M** |

---

## Expected Benefits vs. Baseline CNN

| Improvement         | Mechanism                               | Impact                          |
| ------------------- | --------------------------------------- | ------------------------------- |
| Gating              | Adaptive feature weighting              | ↑ accuracy, ↓ noise sensitivity |
| PCA reduction       | Removes redundant/low-variance features | ↓ overfitting                   |
| LayerNorm           | Per-sample normalization                | ↑ training stability            |
| GELU                | Smooth gradient flow                    | ↑ convergence speed             |
| Label smoothing     | Regularizes output distribution         | ↑ generalization                |
| Smaller fusion head | Fewer parameters to overfit             | ↑ test performance              |

---

## Summary

NeuroSpiral transforms a standard CNN classifier into a **dynamic multimodal system** by:

1. Extracting rich image representations via a pretrained EfficientNet-B0
2. Projecting dimension-reduced handcrafted descriptors through a learned MLP
3. Using a gating network to adaptively weight the contribution of each feature dimension
4. Fusing both modalities through a regularized classification head

This design is conceptually aligned with modern **attention-based multimodal fusion** while remaining computationally lightweight enough to train on small medical datasets.
