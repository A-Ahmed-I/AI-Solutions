# NeuroVive — Unified Parkinson's Disease Detection Platform

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx" alt="ONNX"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Modalities-2-blueviolet" alt="Modalities"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

> **NeuroVive** is a production-ready, multi-modal Parkinson's Disease detection platform that combines two independent AI systems — **NeuroSpiral** (spiral/wave drawing analysis) and **NeuroVox** (voice analysis) — under a single unified FastAPI service.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Models](#models)
- [License](#license)

---

## Overview

Parkinson's Disease (PD) manifests in two well-studied, non-invasive biomarkers:

| Biomarker            | Signal                          | Model                                              |
| -------------------- | ------------------------------- | -------------------------------------------------- |
| **Motor tremor**     | Hand-drawn spiral / wave images | **NeuroSpiral** — EfficientNet-B0 + HOG/LBP fusion |
| **Vocal impairment** | `.wav` voice recordings         | **NeuroVox** — ResNet-18 on Mel-Spectrograms       |

NeuroVive exposes both models through a single FastAPI service with two independent endpoints — allowing clinicians or researchers to use either modality or both together for cross-validation.

---

## Features

| Feature               | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| **Multi-modal**       | Two independent prediction endpoints in one service                  |
| **Async inference**   | Both models run in a `ThreadPoolExecutor` — event loop never blocked |
| **Dual ONNX runtime** | Both models loaded once at startup, shared across all requests       |
| **Auto file cleanup** | Temporary audio files deleted via `BackgroundTasks` after response   |
| **Input validation**  | MIME-type checks for images, extension checks for audio              |
| **Unified schema**    | Both endpoints return the same `PredictionResponse`                  |
| **Swagger UI**        | Interactive docs at `/docs` for both endpoints                       |

---

## System Architecture

```
                        ┌─────────────────────────────────┐
                        │         NeuroVive API           │
                        │         FastAPI + Uvicorn       │
                        └────────────┬────────────┬───────┘
                                     │            │
                    ┌────────────────▼──┐      ┌──▼────────────────┐
                    │  POST /predict    │      │  POST /predict    │
                    │     /image        │      │     /voice        │
                    └────────┬──────────┘      └──────────┬────────┘
                             │                            │
                    ┌────────▼──────────┐      ┌──────────▼────────┐
                    │ NeuroSpiralPred   │      │  NeuroVoxPred     │
                    │ (ONNX — spiral)   │      │ (ONNX — voice)    │
                    └───────────────────┘      └───────────────────┘
                             │                            │
                    ┌────────▼──────────┐      ┌──────────▼────────┐
                    │ EfficientNet-B0   │      │   ResNet-18       │
                    │ + HOG/LBP MLP     │      │ + Mel-Spectrogram │
                    └───────────────────┘      └───────────────────┘
                             │                            │
                             └──────────┬─────────────────┘
                                        │
                               PredictionResponse
                               { label, probability }
```

### Startup Lifecycle

```
App Start
  ├── Create tmp/ directory
  ├── Initialize ThreadPoolExecutor (max_workers=4)
  ├── Load NeuroVoxPredictor     → app.state.voice_model
  └── Load NeuroSpiralPredictor  → app.state.spiral_model

App Shutdown
  └── Shutdown ThreadPoolExecutor (wait=True)
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/A-Ahmed-I/AI-Solutions.git
cd NeuroProduction

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add ONNX checkpoints

Place your exported model files at:

```
checkpoint/
├── spiral_best_model.onnx
└── voice_best_model.onnx
```

> See [NeuroSpiral docs](docs/neurospiral.md) and [NeuroVox docs](docs/neurovox.md) for training and export instructions.

### 3. Run the server

```bash
python -m src.api.main
# Server: http://localhost:8000
# Docs:   http://localhost:8000/docs
```

---

## API Endpoints

### `POST /predict/image`

Classify a spiral or wave drawing image.

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "image=@spiral_drawing.png"
```

```json
{ "label": "HC", "probability": 0.8734 }
```

### `POST /predict/voice`

Classify a voice recording.

```bash
curl -X POST http://localhost:8000/predict/voice \
  -F "audio=@voice_sample.wav"
```

```json
{ "label": "PD", "probability": 0.3102 }
```

**Response schema (both endpoints):**

| Field         | Type     | Values                                                   |
| ------------- | -------- | -------------------------------------------------------- |
| `label`       | `string` | `"HC"` (Healthy Control) or `"PD"` (Parkinson's Disease) |
| `probability` | `float`  | Sigmoid confidence ∈ [0.0, 1.0]                          |

See [`docs/api.md`](docs/api.md) for full reference including error codes.

---

## Project Structure

```
NeuroProduction/
├── README.md
├── requirements.txt
├── checkpoint/
│   ├── spiral_best_model.onnx     ← NeuroSpiral ONNX export
│   └── voice_best_model.onnx      ← NeuroVox ONNX export
├── docs/
│   ├── api.md                     ← Unified API reference
│   ├── architecture.md            ← System design & concurrency model
│   ├── neurospiral.md             ← Spiral model details
│   ├── neurovox.md                ← Voice model details
│   └── inference.md               ← Inference guide for both predictors
└── src/
    ├── api/
    │   ├── endpoint.py            ← FastAPI routes (/predict/image, /predict/voice)
    │   ├── main.py                ← Uvicorn entry point
    │   └── schema.py              ← PredictionResponse Pydantic model
    ├── inference/
    │   ├── neurospiral_predictor.py ← NeuroSpiralPredictor (image → ONNX)
    │   └── neurovox_predictor.py    ← NeuroVoxPredictor (audio → ONNX)
    ├── utils/
    │   └── helper.py              ← lifespan, remove_file, shared utilities
    └── constant/
        └── constant.py            ← Global config (paths, audio params, LBP params)
```

---

## Documentation

| File                                           | Description                                           |
| ---------------------------------------------- | ----------------------------------------------------- |
| [`docs/api.md`](docs/api.md)                   | Full REST API reference — endpoints, errors, examples |
| [`docs/architecture.md`](docs/architecture.md) | System design, concurrency model, lifecycle           |
| [`docs/neurospiral.md`](docs/neurospiral.md)   | Image model — EfficientNet-B0 + HOG/LBP               |
| [`docs/neurovox.md`](docs/neurovox.md)         | Voice model — ResNet-18 + Mel-Spectrogram             |
| [`docs/inference.md`](docs/inference.md)       | Using both predictors directly (without API)          |

---

## Models

| Model           | Backbone                      | Input                          | Task                 |
| --------------- | ----------------------------- | ------------------------------ | -------------------- |
| **NeuroSpiral** | EfficientNet-B0 + HOG/LBP MLP | Spiral/wave image (224×224)    | Image classification |
| **NeuroVox**    | ResNet-18 fine-tuned          | `.wav` audio → Mel-Spectrogram | Audio classification |

Both models output a single raw logit → sigmoid → binary label (`HC` / `PD`).

---

## License

MIT License
