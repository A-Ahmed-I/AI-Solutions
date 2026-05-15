# Training Guide

## Configuration

All hyperparameters are centralized in `src/constant/constant.py`. Edit this file to customize training — no changes needed elsewhere.

```python
# ── Dataset ────────────────────────────────────────────────────────────────
sample_rate       = 22050      # Hz — target resampling rate
min_duration      = 1          # seconds — minimum valid audio length
chunk_duration    = 6          # seconds — length of each audio chunk
overlap_ratio     = 0.1        # 10% overlap between consecutive chunks
n_fft             = 1024       # FFT window size
n_mels            = 40         # number of Mel frequency bands
hop_length        = n_fft // 4 # 256 samples between STFT frames

# ── Quality Filters ─────────────────────────────────────────────────────────
energy_threshold    = 1e-6     # minimum mean-squared energy per chunk
variance_threshold  = 1e-6     # minimum mel-spectrogram variance
silence_db_threshold = -40.0   # minimum 10th-percentile dB

# ── Split ──────────────────────────────────────────────────────────────────
train_ratio  = 0.7             # 70% training
test_ratio   = 0.2             # 20% test (validation = remaining 10%)
random_seed  = 42

# ── Training ───────────────────────────────────────────────────────────────
epochs       = 10
batch_size   = 64

# ── Optimizer ──────────────────────────────────────────────────────────────
lr           = 0.0001
weight_decay = 1e-2            # L2 regularization
```

---

## Optimizer & Scheduler

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-2,
)
```

AdamW decouples weight decay from the gradient update, making regularization more effective than standard Adam with L2 penalty.

### Scheduler: CosineAnnealingWarmRestarts

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,     # restart period in epochs
    T_mult=2,   # period doubles after each restart
)
```

The learning rate follows a cosine decay within each period and resets at warm-restart boundaries. This helps escape local minima and often finds better final optima than a single decay schedule.

```
LR
│▔╲
│  ╲___
│       ▔▔╲___
│              ▔▔▔▔▔╲______
└─────────────────────────── Epochs
  T_0=10    T_0*T_mult=20
```

---

## Loss Function

```python
loss_fn = nn.BCEWithLogitsLoss()
```

Binary Cross-Entropy with Logits combines a Sigmoid and BCE into a numerically stable single operation. The model outputs raw logits; the loss function applies the sigmoid internally.

---

## Training Loop

The `Trainer` class (`src/training/train.py`) manages the full training lifecycle.

### Per-Epoch Flow

```
for each epoch:
    ┌── Training phase (model.train()) ──────────────────────────────┐
    │  for batch in train_loader:                                    │
    │    logits = model(spec)                                        │
    │    loss   = BCEWithLogitsLoss(logits, labels)                  │
    │    optimizer.zero_grad() → loss.backward() → optimizer.step()  │
    │    update Accuracy + F1 metrics                                │
    └────────────────────────────────────────────────────────────────┘
    ┌── Validation phase (model.eval(), no_grad) ─────────────────────┐
    │  for batch in val_loader:                                       │
    │    logits = model(spec)                                         │
    │    loss   = BCEWithLogitsLoss(logits, labels)                   │
    │    update Accuracy + F1 metrics                                 │
    └─────────────────────────────────────────────────────────────────┘

    if val_acc > best_val_acc:
        save checkpoint → best_model.pth

    scheduler.step(val_loss)
    print epoch summary
```

### Checkpointing

The model is saved whenever validation accuracy improves:

```python
if val_acc > self.best_val_acc:
    self.best_val_acc = val_acc
    torch.save(model.state_dict(), checkpoint_path)
```

Only the best weights are retained, preventing overfitting to later epochs when the model begins to memorize training data.

---

## Metrics

Both training and validation phases track:

| Metric   | Implementation                | Threshold |
| -------- | ----------------------------- | --------- |
| Accuracy | `torchmetrics.BinaryAccuracy` | 0.5       |
| F1 Score | `torchmetrics.BinaryF1Score`  | 0.5       |

Metrics are reset at the start of each phase and computed over the full epoch.

---

## Running Training

```bash
# Full pipeline (recommended)
python -m src.main

# Or programmatically
from src.pipeline.pipeline import PipeLine
from src.constant.constant import base_path, checkpoint_path, onnx_path

pipeline = PipeLine(base_path, checkpoint_path, onnx_path)
pipeline.run()
```

### Stage-by-Stage

```python
pipeline.create_metadata(min_duration=1.0)
pipeline.preprocess_audio(metadata)
pipeline.prepare_dataloaders(full_data)
pipeline.train_model(train_loader, val_loader)
pipeline.run_inference(trainer, test_loader)
pipeline.export_onnx(trainer, trainer.model)
```

---

## Post-Training Evaluation

After training, `run_inference` evaluates on the held-out test set and plots a confusion matrix:

```python
trainer.run_inference(test_loader)
# Prints: Accuracy and F1 score
# Shows: Confusion matrix (matplotlib)
```

The confusion matrix is displayed with `ConfusionMatrixDisplay` in `Blues` colormap, showing true positives, false positives, true negatives, and false negatives for `HC` and `PD` classes.

---

## Tips

**Improving generalization:**

- Increase `overlap_ratio` (e.g., 0.25) to generate more chunks per recording
- Reduce `lr` to `5e-5` for slower, more stable convergence
- Add `weight_decay` to 1e-1 if overfitting is observed

**Speeding up training:**

- Enable GPU by ensuring CUDA is available (`torch.cuda.is_available()`)
- Increase `batch_size` to 128 if GPU memory allows
- Reduce `n_mels` to 32 for a smaller spectrogram

**Debugging data issues:**

- Check the `Summary` output from `CreateMetadata.load_metadata()` — a high `Failed files` count suggests audio format or duration issues
- Use `AudioVisualizer.plot_random_sample()` to visually inspect spectrograms before training
