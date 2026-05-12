# NeuroVive

**Multimodal AI System for Early Parkinson's Disease Detection**

NeuroVive analyzes spiral drawings and voice recordings to screen for Parkinson's Disease, combining computer vision and audio deep learning into a single production-ready inference API.

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
  - [NeuroSpiral — Spiral Drawing Classifier](#neurospiral--spiral-drawing-classifier)
  - [NeuroVox — Voice Classifier](#neurovox--voice-classifier)
- [API Reference](#api-reference)
- [Training Results](#training-results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Tech Stack](#tech-stack)

---

## Overview

NeuroVive runs two independent classification models under a single FastAPI application. Each model accepts a different input modality and returns a diagnosis label with a confidence score.

| Modality        | Model       | Input             | Test Accuracy |
| --------------- | ----------- | ----------------- | ------------- |
| Spiral Drawing  | NeuroSpiral | Image (PNG / JPG) | **87.80%**    |
| Voice Recording | NeuroVox    | Audio (WAV)       | **97.80%**    |

**Output labels:**

- `PD` — Parkinson's Disease
- `HC` — Healthy Control

### System Architecture

```
User Upload
     │
     ▼
FastAPI Application
     │
     ├──────────────────────────────────┐
     │                                  │
POST /predict/image              POST /predict/voice
     │                                  │
NeuroSpiral                        NeuroVox
EfficientNet-B0 + HOG + LBP       ResNet-18 + Mel-Spectrogram
     │                                  │
     └──────────────┬───────────────────┘
                    │
                    ▼
         { "label": "PD"|"HC",
           "probability": 0.0–1.0 }
```

**Runtime details:**

- Both ONNX models are loaded at startup via a `lifespan` context manager.
- CPU-bound inference runs in a `ThreadPoolExecutor` (4 workers) to avoid blocking the async event loop.
- Uploaded audio files are saved to `tmp/` and deleted automatically via `BackgroundTasks` after each request.
- GPU acceleration is supported via `CUDAExecutionProvider` with automatic CPU fallback.

---

## Models

### NeuroSpiral — Spiral Drawing Classifier

Classifies hand-drawn spiral or wave images, capturing the fine motor tremors characteristic of Parkinson's Disease.

#### Feature Extraction

NeuroSpiral uses a hybrid approach: a deep CNN backbone extracts high-level visual features, while handcrafted descriptors capture local texture and edge patterns that CNNs may overlook.

| Feature                                   | What It Captures                          |
| ----------------------------------------- | ----------------------------------------- |
| **EfficientNet-B0**                       | High-level visual representations         |
| **HOG** (Histogram of Oriented Gradients) | Edge and shape patterns                   |
| **LBP** (Local Binary Pattern)            | Texture and tremor-related micro-patterns |

#### Model Architecture

```
Input Image (224×224)
       │
       ├───────────────────────┐
       │                       │
  EfficientNet-B0         HOG + LBP
  (1280-dim features)    (512 → 128-dim)
       │                       │
       └──────────┬────────────┘
                  │  Concatenate (1280 + 128)
                  ▼
         Fusion MLP Head
         1408 → 512 → 256 → 1
                  │
               Sigmoid
                  │
           PD / HC + Score
```

#### Training Configuration

| Parameter     | Value                                         |
| ------------- | --------------------------------------------- |
| Backbone      | EfficientNet-B0                               |
| Optimizer     | AdamW                                         |
| Learning Rate | `1e-4`                                        |
| Weight Decay  | `1e-5`                                        |
| Epochs        | 15 (best checkpoint at epoch 13)              |
| Batch Size    | 16                                            |
| LR Scheduler  | Linear warmup (5 epochs) → CosineAnnealing    |
| Loss Function | BCEWithLogitsLoss                             |
| Augmentation  | 15× per image — rotation, affine, noise, flip |

---

### NeuroVox — Voice Classifier

Classifies voice recordings of patients sustaining vowels or reading sentences, detecting vocal biomarkers of Parkinson's Disease.

#### Audio Preprocessing Pipeline

```
WAV File
   │
   ▼
Resample → 22,050 Hz, mono
   │
   ▼
Pad / Truncate → 6 seconds
   │
   ▼
Mel-Spectrogram (n_mels=40, n_fft=1024)
   │
   ▼
Convert to dB scale
   │
   ▼
Tensor shape: (1, 1, 40, T) → Model input
```

#### Model Architecture

```
Input Mel-Spectrogram (1, 40, T)
          │
   ResNet-18 Backbone
   (first conv modified for 1-channel input)
          │
   Classifier Head
   512 → 256 → BatchNorm → SiLU → Dropout → 1
          │
       Sigmoid
          │
    PD / HC + Score
```

#### Training Configuration

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Backbone      | ResNet-18 (ImageNet pretrained) |
| Optimizer     | AdamW                           |
| Learning Rate | `1e-4`                          |
| Weight Decay  | `1e-2`                          |
| Epochs        | 10 (best checkpoint at epoch 6) |
| Batch Size    | 64                              |
| LR Scheduler  | CosineAnnealingWarmRestarts     |
| Loss Function | BCEWithLogitsLoss               |
| Data Split    | 70% train / 20% test / 10% val  |

#### Audio Augmentations

- Time-stretching (1.1× rate)
- Pitch-shifting
- Gaussian noise injection
- Overlapping 6-second chunks (10% overlap)

---

## API Reference

**Base URL:** `http://localhost:8000`

---

### `POST /predict/image`

Classify a spiral or wave drawing.

**Request**

```
Content-Type: multipart/form-data
Body field:   image (file) — JPEG or PNG
```

**Response**

```json
{
  "label": "PD",
  "probability": 0.8342
}
```

**Error Codes**

| Code  | Reason                               |
| ----- | ------------------------------------ |
| `400` | Image is corrupt or unreadable       |
| `415` | File is not a supported image format |
| `500` | Internal inference error             |

---

### `POST /predict/voice`

Classify a voice recording.

**Request**

```
Content-Type: multipart/form-data
Body field:   audio (file) — WAV only
```

**Response**

```json
{
  "label": "HC",
  "probability": 0.1253
}
```

**Error Codes**

| Code  | Reason                        |
| ----- | ----------------------------- |
| `415` | File is not a `.wav` file     |
| `500` | Processing or inference error |

---

### Response Schema

| Field         | Type     | Description                                           |
| ------------- | -------- | ----------------------------------------------------- |
| `label`       | `string` | `"PD"` or `"HC"`                                      |
| `probability` | `float`  | Sigmoid output, rounded to 4 decimal places (0.0–1.0) |

> **Label threshold note:**
>
> - **NeuroSpiral:** `PD` if probability **< 0.5**, `HC` otherwise
> - **NeuroVox:** `PD` if probability **> 0.5**, `HC` otherwise

---

## Training Results

### NeuroSpiral

| Epoch     | Train Loss | Train Acc  | Val Loss   | Val Acc    | Val F1     |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 5         | 0.2479     | 89.41%     | 0.2482     | 87.36%     | 88.38%     |
| 9         | 0.0779     | 97.03%     | 0.2181     | 91.00%     | 91.41%     |
| **13** ✅ | **0.0197** | **99.42%** | **0.1993** | **93.49%** | **93.58%** |
| 15        | 0.0157     | 99.57%     | 0.2296     | 92.53%     | 92.84%     |

**Test Set Performance**

```
Accuracy : 87.80%
F1 Score : 87.80%
```

---

### NeuroVox

| Epoch    | Train Loss | Train Acc  | Val Loss   | Val Acc    | Val F1     |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 2        | 0.2731     | 92.54%     | 0.2971     | 90.11%     | 88.31%     |
| 5        | 0.0637     | 99.52%     | 0.1335     | 96.70%     | 96.10%     |
| **6** ✅ | **0.0513** | **99.84%** | **0.0865** | **97.80%** | **97.37%** |
| 10       | 0.0293     | 100.00%    | 0.0809     | 96.70%     | 96.10%     |

**Dataset Summary**

```
Total valid audio files : 1,274
  PD samples            :   663
  HC samples            :   611
  Rejected / failed     :   404
```

---

## Project Structure

```
AI-Solution/
├── README.md                ← Overview of the whole system
├── LICENSE
├── .gitignore
│
├── NeuroVox/                ← Voice model service
│   ├── README.md
│   ├── requirements.txt
│   ├── docs/
│   └── src/
│
├── NeuroSpiral/             ← Image model service
│   ├── README.md
│   ├── requirements.txt
│   ├── docs/
│   └── src/
│
└── NeuroProduction/         ← Unified inference API
    ├── README.md
    ├── requirements.txt
    ├── checkpoint/
    ├── docs/
    └── src/
```

---

## Installation & Usage

### 1. Install Dependencies

```bash
pip install fastapi uvicorn onnxruntime opencv-python \
            scikit-image librosa numpy
```

### 2. Place ONNX Checkpoints

```
./NeuroProduction/checkpoint/
├── spiral_best_model.onnx
└── voice_best_model.onnx
```

### 3. Start the Server

```bash
python -m ./NeuroProduction/src/api/main.py
```

The API will be available at `http://0.0.0.0:8000`.

### 4. Send a Prediction Request

```bash
# Spiral drawing
curl -X POST "http://localhost:8000/predict/image" \
     -F "image=@spiral_drawing.png"

# Voice recording
curl -X POST "http://localhost:8000/predict/voice" \
     -F "audio=@patient_voice.wav"
```

### 5. Train the Models (Optional)

```bash
# Train NeuroSpiral
python ./NeuroSpiral/src/pipeline/main.py

# Train NeuroVox
python ./NeuroVox/src/pipeline/main.py
```

---

## Tech Stack

| Category          | Library                    |
| ----------------- | -------------------------- |
| API Framework     | FastAPI, Uvicorn           |
| Deep Learning     | PyTorch, timm              |
| Inference Runtime | ONNX Runtime               |
| Image Processing  | OpenCV, scikit-image       |
| Audio Processing  | Librosa                    |
| Data              | Polars, NumPy              |
| Metrics           | torchmetrics, scikit-learn |
| Augmentation      | Albumentations             |

---

_Built for clinical research and early-stage Parkinson's Disease screening support._
