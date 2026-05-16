# Inference Guide

## Overview

NeuroVox uses **ONNX Runtime** for production inference. The `NeuroVoxPredictor` class wraps the exported `.onnx` model and handles all preprocessing internally — you pass in a raw `.wav` file path and receive a label and probability.

```
WAV path → load & resample → normalize length → Mel-Spectrogram → ONNX forward pass → sigmoid → label + probability
```

---

## Prerequisites

Make sure the model has been exported to ONNX before running inference. This happens automatically at the end of the training pipeline, or you can export manually:

```python
from src.pipeline.pipeline import PipeLine
from src.constant.constant import base_path, checkpoint_path, onnx_path

pipeline = PipeLine(base_path, checkpoint_path, onnx_path)
pipeline.export_onnx(trainer, trainer.model)
```

The ONNX file is saved to:

```
checkpoint/best_model.onnx
```

---

## `NeuroVoxPredictor` Class

**File:** `src/inference/predictor.py`

### Constructor

```python
predictor = NeuroVoxPredictor(model_path="checkpoint/best_model.onnx")
```

On initialization:

- Creates an `onnxruntime.InferenceSession` with the configured execution providers
- Caches input/output node names from the model graph
- Sets `target_length = duration × sample_rate` (6 × 22,050 = 132,300 samples)

Raises `RuntimeError` if the ONNX file cannot be loaded.

### `predict(audio_path)`

```python
label, probability = predictor.predict("recording.wav")
```

**Parameters:**

| Parameter    | Type  | Description                             |
| ------------ | ----- | --------------------------------------- |
| `audio_path` | `str` | Path to a `.wav` file (any sample rate) |

**Returns:** `Tuple[str, float]`

| Value         | Type    | Description                                           |
| ------------- | ------- | ----------------------------------------------------- |
| `label`       | `str`   | `"HC"` or `"PD"`                                      |
| `probability` | `float` | Sigmoid confidence score, rounded to 4 decimal places |

Returns `("Error", 0.0)` if audio loading fails.

---

## Preprocessing Pipeline (Inside `predict`)

### Step 1 — Load Audio

```python
audio, _ = librosa.load(audio_path, sr=22050, mono=True)
```

Audio is resampled to 22,050 Hz and forced to mono regardless of the original format.

### Step 2 — Normalize Length

```python
target_length = 6 × 22050 = 132,300 samples

if len(audio) < target_length:
    audio = librosa.util.fix_length(audio, size=target_length)  # zero-pad
else:
    audio = audio[:target_length]                                # truncate
```

All audio is normalized to exactly 6 seconds before feature extraction.

### Step 3 — Mel-Spectrogram

```python
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=22050, n_fft=1024, hop_length=256, n_mels=40
)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
```

Output shape: `(40, ~517)` — 40 Mel bands × ~517 time frames.

### Step 4 — Format for ONNX

```python
spec_tensor = mel_db[np.newaxis, np.newaxis, :, :]  # shape: (1, 1, 40, T)
spec_tensor = spec_tensor.astype(np.float32)
```

### Step 5 — ONNX Inference

```python
logits = session.run([output_name], {input_name: spec_tensor})[0]
logit_value = logits.item()
```

### Step 6 — Sigmoid + Threshold

```python
probability = 1 / (1 + np.exp(-logit_value))
label = "PD" if probability > 0.5 else "HC"
```

---

## Execution Providers

Configured in `src/constant/constant.py`:

```python
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

ONNX Runtime automatically selects the first available provider. If CUDA is not available, it falls back to CPU silently.

To force CPU-only inference:

```python
providers = ["CPUExecutionProvider"]
```

---

## Usage Examples

### Single File

```python
from src.inference.predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")
label, prob = predictor.predict("patient_recording.wav")

print(f"Prediction : {label}")
print(f"Confidence : {prob:.2%}")
# Prediction : HC
# Confidence : 89.21%
```

### Batch Inference Over a Directory

```python
import os
from src.inference.predictor import NeuroVoxPredictor

predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")

results = []
for filename in os.listdir("recordings/"):
    if filename.endswith(".wav"):
        path = os.path.join("recordings/", filename)
        label, prob = predictor.predict(path)
        results.append({"file": filename, "label": label, "probability": prob})

for r in results:
    print(f"{r['file']:<30} → {r['label']}  ({r['probability']:.4f})")
```

### Via the REST API

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@patient_recording.wav"

# Response:
# { "label": "HC", "probability": 0.8921 }
```

---

## ONNX Model Details

| Property      | Value                                     |
| ------------- | ----------------------------------------- |
| Opset version | 11                                        |
| Input name    | `"input"`                                 |
| Input shape   | `(batch_size, 1, 40, T)` — dynamic batch  |
| Output name   | `"outputs"`                               |
| Output shape  | `(batch_size, 1)` — raw logit             |
| Dynamic axes  | batch dimension for both input and output |

---

## Troubleshooting

**`RuntimeError: Failed to load model`**  
Verify the path to the `.onnx` file is correct and the file exists. Re-run training to regenerate it if needed.

**`Error loading audio` / returns `("Error", 0.0)`**  
The WAV file may be corrupt, too short, or in an unsupported encoding. Try resampling with `ffmpeg`:

```bash
ffmpeg -i input.wav -ar 22050 -ac 1 output.wav
```

**Slow inference on CPU**  
Consider using `onnxruntime-gpu` and ensuring CUDA drivers are installed, or reduce `n_mels` and `chunk_duration` for lighter feature extraction.
