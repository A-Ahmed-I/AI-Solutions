<div align="center">

# NeuroVox

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-1.23.2-green?style=for-the-badge&logo=onnx)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128.1-009688?style=for-the-badge&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**AI-Powered Parkinson's Disease Detection from Voice Analysis**

_Advanced deep learning system for early detection of Parkinson's Disease through speech pattern recognition — achieving up to **98.33% accuracy** with less than one minute of training time._

[Overview](#-overview) • [Performance](#-performance) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Models](#-model-architectures) • [API](#-rest-api) • [Configuration](#-configuration-reference)

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Performance](#-performance)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Model Architectures](#-model-architectures)
- [Audio Preprocessing](#-audio-preprocessing)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [REST API](#-rest-api)
- [Python SDK](#-python-sdk)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## Overview

**NeuroVox** is a state-of-the-art deep learning system that detects Parkinson's Disease (PD) through voice analysis. It converts voice recordings into mel-spectrogram images and runs them through CNN-based classifiers to distinguish PD patients from Healthy Control (HC) subjects.

> Voice changes are among the **earliest symptoms** of Parkinson's Disease, often appearing years before motor symptoms. NeuroVox makes non-invasive, early screening accessible to everyone.

### Why NeuroVox?

| Benefit             | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Non-Invasive** | Simple voice recording — no physical examination required    |
| **Fast**         | Results in under 1 second at inference; trains in < 1 minute |
| **Accurate**     | Up to 98.33% accuracy with clinical-grade models             |
| **Accessible**   | Easy-to-use REST API and Python SDK                          |
| **Flexible**     | Runs on both GPU and CPU environments                        |

---

## Performance

Evaluated on **958 samples** (510 PD · 448 HC):

| Model             | Accuracy   | Sensitivity | Specificity | F1 Score   |
| ----------------- | ---------- | ----------- | ----------- | ---------- |
| NeuroVoxCNN       | 95.72%     | 96.47%      | 94.87%      | 0.9597     |
| NeuroVoxRN        | 94.26%     | 97.84%      | 90.18%      | 0.9478     |
| **NeuroVoxTL** ⭐ | **98.33%** | **99.02%**  | **97.54%**  | **0.9845** |

> **→ Use NeuroVoxTL for production.** It trains fastest, performs best, and maintains strong balance between sensitivity and specificity.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       NeuroVox System                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐   │
│  │   REST API   │ ──── │  Predictor   │ ──── │   ONNX   │   │
│  │   (FastAPI)  │      │    Engine    │      │   Model  │   │
│  └──────────────┘      └──────────────┘      └──────────┘   │
│         │                      │                     │      │
│  ┌──────▼──────┐      ┌────────▼────────┐   ┌────────▼──┐   │
│  │   Upload    │      │  Preprocessing  │   │ Inference │   │
│  │   Handler   │      │   (Librosa)     │   │  Runtime  │   │
│  └─────────────┘      └─────────────────┘   └───────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
WAV File → Load & Normalize → Chunk (6s) → Mel-Spectrogram
         → Quality Filter → Standardize → ONNX Inference
         → Sigmoid → Classification → Label + Probability
```

### Project Structure

```
neurovox/
├── data/
│   ├── HC/                          # Healthy Control .wav files
│   └── PD/                          # Parkinson's Disease .wav files
├── checkpoint/                      # Saved model weights (.pth, .onnx)
├── requirements.txt
├── LICENSE
├── README.md
└── src/
    ├── api/
    │   ├── main.py                  # FastAPI application entry point
    │   └── endpoint.py              # REST API endpoints
    ├── constant/
    │   └── constant.py              # All hyperparameters and paths
    ├── data/
    │   ├── custom_data.py           # PyTorch Dataset
    │   ├── data_loader.py           # Train/val/test splits + DataLoaders
    │   └── metadata.py              # Metadata management
    ├── inference/
    │   ├── predictor.py             # Core prediction engine
    │   └── __init__.py
    ├── models/
    │   ├── neurovox_cnn.py          # Custom CNN
    │   ├── neurovox_rn.py           # ResNet-based model
    │   └── neurovox_tl.py           # Transfer learning model
    ├── pipeline/
    │   ├── main.py                  # Entry point
    │   └── pipeline.py              # Pipeline orchestrator
    ├── preprocessing/
    │   └── processing.py            # Audio feature extraction
    ├── training/
    │   └── train.py                 # Training loop
    └── plots/
        └── plot.py                  # Visualization utilities
```

---

## Installation

### Prerequisites

| Requirement  | Version    | Notes                           |
| ------------ | ---------- | ------------------------------- |
| Python       | 3.8+       | Required                        |
| RAM          | 4 GB+      | Required                        |
| CUDA Toolkit | 11.0+      | Optional — for GPU acceleration |
| cuDNN        | Compatible | Optional — for GPU acceleration |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/neurovox.git
cd neurovox

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import librosa; print(librosa.__version__)"
python -c "from src.inference.predictor import NeuroVoxPredictor; print('✅ Ready!')"
```

---

## Quick Start

### Training Pipeline

**1. Prepare your dataset**

```
data/
├── HC/          # Healthy Control — .wav files
└── PD/          # Parkinson's Disease — .wav files
```

Audio requirements: WAV format, any sample rate (auto-resampled to 22,050 Hz), any duration (auto-chunked into 6-second segments).

**2. Run the full pipeline**

```bash
# Preprocess → Train → Evaluate
python src/pipeline/main.py
```

**3. Outputs**

| Location      | Contents                             |
| ------------- | ------------------------------------ |
| `checkpoint/` | Saved `.pth` and `.onnx` model files |

---

### Inference (30-Second Demo)

```python
from src.inference.predictor import NeuroVoxPredictor

# Initialize the predictor
predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")

# Run prediction
label, probability = predictor.predict("your_audio.wav")

# Display results
print(f"Diagnosis:  {label}")
print(f"Confidence: {probability:.2%}")
```

**Example output:**

```
Diagnosis:  PD
Confidence: 94.32%
```

---

## Model Architectures

### NeuroVoxCNN — Lightweight Custom CNN

Best for edge deployment or memory-constrained environments.

```
Input (1 × Time × 40 mel bands)
  → Block 1: Conv2D(3×3) → BN → ReLU → MaxPool(3)   [1→32 ch]
  → Block 2: Conv2D(2×2) → BN → ReLU → MaxPool(2)   [32→64 ch]
  → Block 3: Conv2D(3×3) → BN → ReLU → MaxPool(3)   [64→128 ch]
  → Global Average Pool
  → FC(128→64) → BN → ReLU → Dropout(0.5)
  → FC(64→32)  → BN → ReLU → Dropout(0.5)
  → FC(32→1)   → Sigmoid
```

~85K parameters · 30 epochs · ~15 min training

**Confusion Matrix:**

```
              Pred HC   Pred PD
Actual HC       425        23
Actual PD        18       492
```

---

### NeuroVoxRN — ResNet-Based

Best when minimizing false negatives is the top priority (highest sensitivity at **97.84%**).

```
Input (1 × Time × 40)
  → Initial Conv2D(3×3) → BN → ReLU
  → Residual Block 1 → MaxPool(2)   [64 ch]
  → Residual Block 2 → MaxPool(2)  [128 ch]
  → Residual Block 3 → MaxPool(2)  [512 ch]
  → Global Average Pool
  → Dropout(0.5) → FC(512→256) → BN → ReLU → Dropout(0.5)
  → FC(256→1) → Sigmoid
```

~275K parameters · 30 epochs · ~2 min training

**Confusion Matrix:**

```
              Pred HC   Pred PD
Actual HC       404        44
Actual PD        11       499
```

> Only **11 false negatives** — the best among all three models.

---

### NeuroVoxTL — Transfer Learning ⭐ (Recommended)

Best overall. ResNet18 backbone pretrained on ImageNet, fine-tuned with a custom classifier head.

```
Input (1 × Time × 40)
  → Modified Conv1: Conv2D(3×7), 1→64 ch (grayscale-adapted)
  → ResNet18 Backbone (frozen):
       Layer 1: 64 ch
       Layer 2: 128 ch
       Layer 3: 256 ch
       Layer 4: 512 ch  + skip connections throughout
  → Feature vector: 512 dims
  → FC(512→256) → BN → SiLU → Dropout(0.5)
  → FC(256→1)   → Sigmoid
```

~11M parameters (backbone frozen) · 10 epochs · < 1 min training

**Confusion Matrix:**

```
              Pred HC   Pred PD
Actual HC       437        11
Actual PD         5       505
```

> Strong start from pretrained weights — epoch 1 already achieves **91.87%** validation accuracy.

### Model Specifications (Inference)

| Property               | Value                              |
| ---------------------- | ---------------------------------- |
| **Runtime**            | ONNX 1.23.2                        |
| **Input Shape**        | (batch, 1, 40, 264)                |
| **Input Type**         | Float32                            |
| **Output Shape**       | (batch, 1)                         |
| **Output Type**        | Float32 (Sigmoid)                  |
| **Decision Threshold** | 0.5 — above → PD, at or below → HC |
| **Model Size**         | ~2 MB                              |
| **Inference Time**     | < 100ms (GPU) · < 500ms (CPU)      |

---

## Audio Preprocessing

Every audio file goes through this pipeline before training or inference:

```
1. Load & Normalize
   - Resample to 22,050 Hz
   - Convert stereo → mono

2. Chunk
   - Split into 6-second segments with 10% overlap
   - Pad final chunk if needed

3. Extract Mel-Spectrogram
   - FFT size: 1,024  |  Hop: 512  |  Mel bands: 40
   - Apply log scaling → shape (40, T)

4. Quality Filter
   - Drop chunks below energy, variance, or silence thresholds

5. Standardize
   - Zero mean, unit variance
```

### Data Augmentation (Training Only)

| Technique       | Range        |
| --------------- | ------------ |
| Time stretch    | ±10%         |
| Pitch shift     | ±2 semitones |
| Noise injection | SNR 20–40 dB |
| Volume          | ±10 dB       |

---

## Training

All three models share the same training configuration for a fair comparison:

```python
loss_fn   = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
```

Key behaviors:

- Best model checkpoint saved based on validation accuracy
- Automatic CUDA detection with CPU fallback
- Dropout (0.5) + weight decay for regularization
- Per-epoch tracking of loss, accuracy, and F1 score

---

## Evaluation

For medical screening, **sensitivity** (catching true PD cases) is the primary metric, while specificity keeps false positives manageable.

```
Accuracy    = (TP + TN) / Total
Sensitivity = TP / (TP + FN)   ← minimize missed PD cases
Specificity = TN / (TN + FP)   ← reduce unnecessary follow-ups
F1 Score    = harmonic mean of Precision and Recall
```

---

## REST API

The FastAPI-based REST API provides the easiest integration for production systems.

### Start the Server

```bash
python -m src.api.main
```

Server starts at: **`http://localhost:8000`**

Interactive docs available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /predict`

Upload a WAV audio file and receive a PD prediction.

**Request:**

```http
POST /predict HTTP/1.1
Content-Type: multipart/form-data

file: <binary-wav-file>
```

**Response:**

```json
{
  "label": "PD",
  "probability": 0.9432
}
```

**Status Codes:**

| Code                        | Meaning                             |
| --------------------------- | ----------------------------------- |
| `200 OK`                    | Successful prediction               |
| `400 Bad Request`           | Invalid file format or missing file |
| `500 Internal Server Error` | Model inference error               |

### Usage Examples

**cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_sample.wav"
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("audio_sample.wav", "rb")}
)
result = response.json()
print(f"Label: {result['label']}")
print(f"Probability: {result['probability']}")
```

---

## Python SDK

### `NeuroVoxPredictor` Class Reference

```python
class NeuroVoxPredictor:
    """Main inference engine for Parkinson's Disease detection."""

    def __init__(self, model_path: str):
        """
        Initialize the predictor with an ONNX model.

        Args:
            model_path (str): Path to the ONNX model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX Runtime initialization fails
        """

    def preprocess(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file into model input format.

        Args:
            audio_path (str): Path to WAV audio file

        Returns:
            np.ndarray: Preprocessed tensor of shape (1, 1, 40, time_steps)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is invalid
        """

    def predict(self, audio_path: str) -> tuple[str, float]:
        """
        Predict Parkinson's Disease from audio file.

        Args:
            audio_path (str): Path to WAV audio file

        Returns:
            tuple[str, float]: (label, probability)
                - label: "PD" (Parkinson's) or "HC" (Healthy Control)
                - probability: Confidence score [0.0 – 1.0]

        Example:
            >>> predictor = NeuroVoxPredictor("checkpoint/best_model.onnx")
            >>> label, prob = predictor.predict("audio.wav")
            >>> print(f"{label}: {prob:.2%}")
            PD: 94.32%
        """
```

---

## Configuration Reference

All settings live in `src/constant/constant.py`.

### Audio Parameters

| Parameter          | Default | Notes                        |
| ------------------ | ------- | ---------------------------- |
| `SAMPLE_RATE`      | `22050` | Sampling frequency in Hz     |
| `CHUNK_DURATION`   | `6`     | Audio clip length in seconds |
| `CHUNK_OVERLAP`    | `0.10`  | 10% overlap between chunks   |
| `N_MELS`           | `40`    | Mel-frequency bands          |
| `N_FFT`            | `1024`  | FFT window size (power of 2) |
| `HOP_LENGTH`       | `512`   | Samples between frames       |
| `USE_AUGMENTATION` | `True`  | Enable training augmentation |

### Training Parameters

| Parameter       | Default         | Notes                                          |
| --------------- | --------------- | ---------------------------------------------- |
| `MODEL_TYPE`    | `neurovox_tl`   | `neurovox_cnn` · `neurovox_rn` · `neurovox_tl` |
| `EPOCHS`        | `30` (TL: `10`) |                                                |
| `BATCH_SIZE`    | `64`            | Reduce to 32/16 if out of memory               |
| `LEARNING_RATE` | `1e-4`          |                                                |
| `WEIGHT_DECAY`  | `0.01`          |                                                |
| `NUM_WORKERS`   | `4`             |                                                |

### Dataset Split

| Parameter     | Default |
| ------------- | ------- |
| `TRAIN_RATIO` | `0.70`  |
| `VAL_RATIO`   | `0.10`  |
| `TEST_RATIO`  | `0.20`  |

### Hardware Acceleration

```python
# Priority order in constant.py
providers = [
    "CUDAExecutionProvider",   # GPU (if available)
    "CPUExecutionProvider"     # CPU (fallback)
]

# Force CPU-only
providers = ["CPUExecutionProvider"]

# Verify GPU availability
import onnxruntime as ort
print(ort.get_available_providers())
```

---

## Troubleshooting

### CUDA / GPU Issues

**"CUDA provider not available"**

```bash
nvidia-smi   # Check CUDA installation
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
pip install onnxruntime-gpu==1.23.2   # Install GPU version if needed
```

**"CUDA out of memory"** — Reduce `BATCH_SIZE` (try 32 or 16), switch to `neurovox_cnn` (smallest model), or run `torch.cuda.empty_cache()` between runs.

---

### Audio Issues

**"Audio file not supported"** — Convert to WAV using FFmpeg:

```bash
ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
```

Or using Python:

```python
from pydub import AudioSegment
AudioSegment.from_file("input.mp3").export("output.wav", format="wav")
```

**Audio loading errors** — Confirm files are uncompressed WAV, duration > 1 second, and not corrupted.

---

### Training Issues

**Low accuracy** — Check for class imbalance and apply class weights if needed. Try reducing the learning rate to `1e-5` or enabling augmentation.

**Loss is NaN** — Enable gradient clipping (`GRADIENT_CLIP_VALUE = 1.0`) and verify your labels are `0`/`1` integers.

**Slow preprocessing** — Increase `NUM_WORKERS`, move data to an SSD, or pre-cache spectrograms to disk.

---

### Common Errors

| Error                | Cause                       | Fix                                         |
| -------------------- | --------------------------- | ------------------------------------------- |
| `FileNotFoundError`  | Missing model or audio file | Check file paths                            |
| `RuntimeError: ONNX` | Model failed to load        | Verify ONNX file integrity                  |
| `ValueError: shape`  | Wrong audio format          | Convert to mono WAV                         |
| `CUDA out of memory` | GPU overload                | Switch to CPU provider or reduce batch size |

---

## Contributing

We welcome contributions! Here's how to get started:

```bash
# Fork and clone
git clone https://github.com/yourusername/neurovox.git
cd neurovox

# Create a feature branch
git checkout -b feature/your-feature-name

# Install dev dependencies
pip install -r requirements-dev.txt

# Make your changes, then run tests and linting
pytest tests/
flake8 src/
black src/

# Push and open a Pull Request
git push origin feature/your-feature-name
```

**Good areas to contribute:** new architectures, preprocessing improvements, additional evaluation metrics, deployment utilities, and documentation.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
✅ Commercial use    ✅ Modification
✅ Distribution      ✅ Private use
❌ Liability         ❌ Warranty
```


