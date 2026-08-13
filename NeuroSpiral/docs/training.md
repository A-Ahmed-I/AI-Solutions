# Training

This document covers the training configuration, loss function, optimiser, learning rate schedule, checkpointing strategy, and training loop implementation.

---

## Overview

Training is orchestrated by the `Trainer` class, which wraps the model, loss function, optimiser, and scheduler into a single `train_model()` method. Per epoch, the trainer:

1. Runs a full training pass (`run_epoch(train=True)`)
2. Runs a validation pass (`run_epoch(train=False)`)
3. Saves a checkpoint if validation F1 has improved
4. Steps the LR scheduler
5. Logs all metrics to a history dictionary

On completion, `plot_training_history()` renders loss, accuracy, and F1 curves.

---

## Loss Function

```python
nn.BCEWithLogitsLoss()
```

Fuses the sigmoid activation and binary cross-entropy into one numerically stable operation using the log-sum-exp trick. Always preferred over `sigmoid` + `BCELoss` for binary classification.

### Label Smoothing

To reduce overconfident predictions on the small dataset, soft labels are used:

```python
smooth = 0.1
label_smooth = label * (1 - smooth) + 0.5 * smooth
loss = loss_fn(logits, label_smooth.unsqueeze(1).float())
```

With `smooth=0.1`, a ground-truth `1.0` becomes `0.95` and a ground-truth `0.0` becomes `0.05`. This prevents the model from driving logits to ±∞ and improves calibration.

---

## Optimiser

```python
torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)
```

| Hyperparameter | Value  | Rationale                                                  |
| -------------- | ------ | ---------------------------------------------------------- |
| Learning rate  | `1e-4` | Moderate LR suitable for fine-tuning a pretrained backbone |
| Weight decay   | `1e-4` | L2 regularisation coefficient to reduce overfitting        |

AdamW decouples weight decay from the gradient update step — this is the mathematically correct formulation for L2 regularization with adaptive optimisers (unlike standard Adam with `l2_reg`).

---

## Learning Rate Schedule

A two-phase schedule: **linear warmup** → **cosine annealing**.

```
LR (×10⁻⁴)
1.0  │                  ●
     │              ●       ●
0.8  │           ●              ●
     │       ●                      ●
0.4  │    ●                              ●
     │  ●                                    ●
0.0  │●                                           ●
     └──────────────────────────────────────────────── Epoch
       1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
       ◄─── Warmup ────►◄──────────── Cosine Annealing ──────────►
```

### Phase 1 — Linear Warmup (Epochs 1–5)

```python
LinearLR(
    optimizer,
    start_factor=0.01,   # LR = 1e-4 × 0.01 = 1e-6 at epoch 1
    end_factor=1.0,      # LR = 1e-4 at epoch 5
    total_iters=5,
)
```

Warmup prevents large gradient updates during the first epochs when model weights (especially the randomly initialized fusion head) are far from their optimal values.

### Phase 2 — Cosine Annealing (Epochs 6–15)

```python
CosineAnnealingLR(
    optimizer,
    T_max=10,   # one full cosine cycle over 10 epochs
)
```

Gradually decays LR, allowing the model to settle into a smooth loss basin. The low LR in the final epochs (1e-5 → 0 by epoch 15) is particularly effective for fine-grained weight adjustments.

### SequentialLR

```python
SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[5],   # switch at epoch 5
)
```

### Observed LR Trajectory

| Epoch | LR       |
| ----- | -------- |
| 1     | 0.000021 |
| 2     | 0.000041 |
| 3     | 0.000060 |
| 4     | 0.000080 |
| 5     | 0.000100 |
| 6     | 0.000098 |
| 7     | 0.000090 |
| 8     | 0.000079 |
| 9     | 0.000065 |
| 10    | 0.000050 |
| 11    | 0.000035 |
| 12    | 0.000021 |
| 13    | 0.000010 |
| 14    | 0.000002 |
| 15    | 0.000000 |

---

## Training Loop

### Per-Epoch Execution

```python
train_loss, train_acc, train_f1 = run_epoch(train=True,  dataloader=train_dataloader)
val_loss,   val_acc,   val_f1   = run_epoch(train=False, dataloader=val_dataloader)
```

`run_epoch` sets the model to `.train()` or `.eval()` mode and uses `torch.enable_grad()` / `torch.no_grad()` contexts accordingly.

---

### Training Step

```python
logits = model(images, math_features)          # (B, 1)

smooth = 0.1
label_smooth = label * (1 - smooth) + 0.5 * smooth
loss = loss_fn(logits, label_smooth.unsqueeze(1).float())

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

### Metrics

Accuracy and F1 are computed using `torchmetrics` with a fixed threshold:

```python
BinaryAccuracy(threshold=0.5)
BinaryF1Score(threshold=0.5)
```

Metrics are **reset at the start of each epoch** to prevent accumulation across batches. Both training and validation metrics are tracked.

> **Note:** The `threshold=0.5` here is for tracking training progress only. The final test evaluation uses a threshold of **0.46**, tuned on the validation set after training completes.

---

## Checkpointing

The best model is saved whenever **validation F1 improves**:

```python
if val_f1 > self.best_val_f1:
    self.best_val_f1 = val_f1
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch+1} (Val F1: {val_f1:.2f}%)")
```

Only the `state_dict` is saved (weights only, not the full optimizer state), keeping file size minimal. The checkpoint is loaded for testing and ONNX export.

**Best checkpoint:** Epoch 13 — Val F1: **97.03%**, Val Acc: **96.94%**

---

## Hardware

```python
self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

Training was performed on a **Kaggle P100 GPU**. Total training time: approximately **25–40 minutes** for 15 epochs (varies with GPU queue time).

---

## Training Configuration Summary

| Parameter             | Value                                       |
| --------------------- | ------------------------------------------- |
| Epochs                | 15                                          |
| Train batch size      | 16                                          |
| Validation batch size | 8                                           |
| Loss                  | `BCEWithLogitsLoss` + label smoothing (0.1) |
| Optimiser             | `AdamW`                                     |
| Base LR               | `1e-4`                                      |
| Weight decay          | `1e-4`                                      |
| Warmup epochs         | 5                                           |
| Cosine T_max          | 10                                          |
| Checkpoint criterion  | Maximum validation F1                       |
| Best epoch            | 13                                          |
| Best val F1           | 97.03%                                      |

---

## Training Visualization

`plot_training_history()` generates three side-by-side plots after training:

| Plot     | X-axis | Y-axis    | Lines         |
| -------- | ------ | --------- | ------------- |
| Loss     | Epoch  | BCE Loss  | Train vs. Val |
| Accuracy | Epoch  | % Correct | Train vs. Val |
| F1 Score | Epoch  | F1 (%)    | Train vs. Val |

These plots are useful for identifying:

- **Overfitting:** growing gap between train and val curves
- **Underfitting:** both curves plateau at low performance
- **Optimal stopping point:** where val metric peaks before degrading
