# Data Pipeline

## Overview

The data pipeline transforms raw `.wav` audio files into normalized Mel-Spectrogram tensors ready for CNN training. It consists of four sequential stages:

```
Raw WAV → Validation → Chunking → Augmentation → Mel-Spectrogram → Dataset
```

---

## Stage 1 — Metadata Creation (`CreateMetadata`)

**File:** `src/data/metadata.py`

Before any audio is loaded into memory, `CreateMetadata` scans the dataset directory, validates each file, and builds a structured Polars DataFrame.

### Validation Checks

Each file must pass all four checks to be included:

| Check              | Method                            | Criteria                                |
| ------------------ | --------------------------------- | --------------------------------------- |
| File integrity     | `soundfile.read` / `librosa.load` | File must be readable                   |
| Non-silence        | Energy check                      | `max(abs(signal)) ≥ 1e-6`               |
| Minimum duration   | Length check                      | Duration ≥ `min_duration` (default: 1s) |
| Numerical validity | NaN/Inf check                     | All samples must be finite              |

### Output

```python
pl.DataFrame(schema=["Path", "Label"])
# Label ∈ {"PD", "HC"}
```

### Example Summary

```
Valid files : 1,274
Failed files: 404
PD samples  : 663
HC samples  : 611
```

---

## Stage 2 — Preprocessing (`PreProcessing`)

**File:** `src/preprocessing/processing.py`

This is the most computationally intensive stage. Processing is parallelized across CPU cores using `ThreadPoolExecutor`.

### Audio Loading

```python
librosa.load(path, sr=22050, mono=True)
```

Audio is always resampled to 22,050 Hz and converted to mono. Results are cached with `@lru_cache` to avoid redundant disk reads.

### Chunking

Long recordings are split into fixed-length overlapping windows:

| Parameter           | Value           |
| ------------------- | --------------- |
| `chunk_duration`    | 6 seconds       |
| `overlap_ratio`     | 0.10 (10%)      |
| `chunk_len`         | 132,300 samples |
| `hop_length_chunks` | 119,070 samples |

Short recordings (< `min_duration`) are zero-padded to `chunk_len`.

A tail chunk is added if the remaining audio after the last full window is ≥ `chunk_len / 2`.

### Augmentation

For each valid chunk, three augmented variants are generated:

| Augmentation   | Method                                   | Notes                         |
| -------------- | ---------------------------------------- | ----------------------------- |
| Original       | —                                        | Always included               |
| Time-stretch   | `librosa.effects.time_stretch(rate=1.1)` | Only if chunk > 5,000 samples |
| Pitch-shift    | `librosa.effects.pitch_shift`            | Only if chunk > 100 samples   |
| Gaussian noise | `audio + 0.005 × N(0,1)`                 | Always applied                |

Each augmented variant is then padded/truncated to exactly `chunk_len`.

### Mel-Spectrogram Extraction

```python
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=22050, n_fft=1024, hop_length=256, n_mels=40
)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
```

Output shape per chunk: `(40, T)` where `T ≈ 517` time frames for a 6-second chunk.

### Quality Filtering

Each Mel-Spectrogram is validated before being added to the dataset:

| Filter     | Criteria                       | Purpose                                |
| ---------- | ------------------------------ | -------------------------------------- |
| Energy     | `mean(chunk²) ≥ 1e-6`          | Remove silent chunks                   |
| Silence dB | `percentile(mel, 10) ≥ -40 dB` | Remove near-silent spectrograms        |
| Variance   | `var(mel) ≥ 1e-6`              | Remove flat/uninformative spectrograms |

---

## Stage 3 — Dataset (`AudioData`)

**File:** `src/data/custom_data.py`

`AudioData` wraps the list of `(spectrogram, label)` tuples into a PyTorch `Dataset`.

### Per-sample transformation

```python
# spectrogram: np.ndarray (H, W) → torch.float32 (1, H, W)
tensor = torch.as_tensor(spec, dtype=torch.float32).unsqueeze(0)

# label: str → int → torch.long
label_int = {"HC": 0, "PD": 1}[label]
```

The channel dimension (1) is added to make spectrograms compatible with 2D convolution layers that expect `(B, C, H, W)` inputs.

---

## Stage 4 — Data Splitting (`Loader`)

**File:** `src/data/data_loader.py`

Data is split using stratified sampling to preserve class balance across all three sets.

### Split Ratios

| Set        | Ratio | Purpose                         |
| ---------- | ----- | ------------------------------- |
| Train      | 70%   | Model training                  |
| Test       | 20%   | Held-out evaluation             |
| Validation | 10%   | Epoch monitoring, checkpointing |

### DataLoader Configuration

| Loader     | Shuffle | Batch Size |
| ---------- | ------- | ---------- |
| Train      | ✅ Yes  | 64         |
| Test       | ❌ No   | 64         |
| Validation | ❌ No   | 64         |

---

## Visualization

The `AudioVisualizer` class (in `src/visualization/`) lets you inspect random samples from the metadata:

```python
from src.visualization.visualizer import AudioVisualizer

viz = AudioVisualizer(metadata)
viz.plot_random_sample()   # displays waveform + mel-spectrogram side by side
```

This produces a two-panel figure showing the raw waveform and its log-scaled Mel-Spectrogram for a randomly chosen sample, labeled by class.
