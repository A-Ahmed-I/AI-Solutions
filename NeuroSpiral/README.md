# NeuroSpiral

> **Parkinson's Disease Detection from Spiral & Wave Drawings using Hybrid Deep Learning**

---

## Overview

NeuroSpiral is a machine learning system that detects **Parkinson's Disease (PD)** from hand-drawn spiral and wave sketches. It combines a pretrained **EfficientNet-B0** CNN with handcrafted **HOG + LBP** texture features in a hybrid fusion architecture with an **adaptive gating mechanism**, achieving **97.03% F1** on the validation set and **90.91% F1** on the held-out test set.

The system is packaged as a **REST API** (FastAPI + ONNX Runtime) that accepts an image upload and returns a prediction label (`PD` / `HC`) with a probability score.

---

## Features

- **Hybrid Architecture** — CNN image features fused with handcrafted HOG + LBP descriptors via adaptive gating
- **Adaptive Gating** — The model learns _when_ to trust handcrafted features, suppressing them when noisy
- **Production-Ready API** — FastAPI endpoint with async inference, validated input handling, and structured JSON responses
- **ONNX Export** — Model exported to ONNX for fast, framework-agnostic deployment
- **Full Training Pipeline** — Data loading → preprocessing → augmentation → feature extraction → training → evaluation → export
- **Data Augmentation** — ×6 expansion per sample (rotation, affine, Gaussian noise, brightness)
- **Warmup + Cosine Annealing LR** — Two-phase scheduler for stable convergence on small datasets

---

## Architecture Overview

```
Input Image (B, 3, 224, 224)       Math Features (B, K)
         │                                  │
         ▼                                  ▼
  EfficientNet-B0                      LayerNorm
    Backbone                               │
         │                            MLP Projection
         ▼                            K → 256 → 128
   CNN Features                            │
    (B, 1280)                       Math Embedding
         │                             (B, 128)
         └──────────┬──────────────────────┘
                    │
                    ▼
         Concatenation (B, 1408)
                    │
                    ▼
            Gate Network
          1408 → 128 (Sigmoid)
                    │
        Gate applied to Math Embedding
        math_feat = math_feat × gate
                    │
                    ▼
      Final Concat (B, 1408)
                    │
                    ▼
             Fusion Head
          1408 → 256 → 64 → 1
                    │
                  Logit
                    │
         ┌──────────────────────┐
         │  prob ≥ 0.46 → HC    │
         │  prob < 0.46 → PD    │
         └──────────────────────┘
```

---

## Project Structure

```
NeuroSpiral/
├── README.md                      ← Project overview (this file)
├── requirements.txt               ← Python dependencies
│
├── src/
│   ├── pipeline/
│   │   ├── main.py                ← Entry point for pipeline execution
│   │   └── pipeline.py            ← End-to-end training pipeline
│   │
│   ├── inference/
│   │   └── predictor.py           ← ONNX inference class
│   │
│   ├── augmentation/
│   │   └── augmented.py           ← Data augmentation utilities
│   │
│   ├── data/
│   │   ├── metadata.py            ← Dataset metadata handling
│   │   └── custom_data.py         ← Custom PyTorch Dataset
│   │
│   ├── feature_extraction/
│   │   └── handcrafted.py         ← HOG + LBP feature extraction
│   │
│   ├── processing/
│   │   └── processing.py          ← Image preprocessing pipeline
│   │
│   ├── training/
│   │   └── train.py               ← Model training loop
│   │
│   ├── model/
│   │   └── neurospiral.py         ← Model architecture definition
│   │
│   ├── utils/
│   │   └── helper.py              ← Utility functions
│   │
│   ├── api/
│   │   ├── main.py                ← FastAPI app entry point
│   │   ├── endpoint.py            ← /predict route implementation
│   │   └── schema.py              ← Pydantic request/response models
│   │
│   ├── constant/
│   │   └── constant.py            ← All hyperparameters, paths, and configs
│   │
│   └── __init__.py
│
└── docs/
    ├── architecture.md            ← Model & system design
    ├── data_pipeline.md           ← Data loading, preprocessing, augmentation
    ├── training.md                ← Training loop, scheduler, checkpointing
    ├── inference_api.md           ← API usage and ONNX inference
    ├── configuration.md           ← All constants and hyperparameters
    └── results.md                 ← Model performance and metrics
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training; CPU is sufficient for inference)

### 1. Clone the Repository

```bash
git clone https://github.com/A-Ahmed-I/AI-Solutions.git
cd NeuroSpiral
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Training the Model

```bash
python -m src.pipeline.pipeline
```

This runs the full pipeline in order:

1. Loads and preprocesses the Parkinson's drawings dataset (Fourier LPF → Otsu → Morphology)
2. Augments training images (×6 per sample)
3. Extracts HOG + LBP handcrafted features and applies PCA reduction
4. Trains the hybrid classifier for 15 epochs with warmup + cosine annealing
5. Saves the best checkpoint based on validation F1
6. Evaluates on the held-out test set with TTA and threshold tuning
7. Exports the model to ONNX format

### Starting the API Server

```bash
python src/main.py
```

The server starts at `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

### Making a Prediction

**Via `curl`:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "image=@/path/to/spiral_drawing.png"
```

**Via Python `requests`:**

```python
import requests

with open("spiral_drawing.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"image": ("drawing.png", f, "image/png")},
    )

result = response.json()
print(result)
# {"label": "HC", "probability": 0.8342}
```

---

## Example Output

```json
{
  "label": "HC",
  "probability": 0.8342
}
```

| Field         | Type     | Description                                              |
| ------------- | -------- | -------------------------------------------------------- |
| `label`       | `string` | `"HC"` (Healthy Control) or `"PD"` (Parkinson's Disease) |
| `probability` | `float`  | Sigmoid probability ∈ [0, 1]. Values ≥ 0.46 → HC         |

---

## Results Summary

| Metric      | Validation (Best — Epoch 13) | Test Set |
| ----------- | ---------------------------- | -------- |
| Accuracy    | 96.94%                       | 90.24%   |
| F1 Score    | 97.03%                       | 90.91%   |
| AUC-ROC     | —                            | 93.57%   |
| Sensitivity | —                            | 95.24%   |
| Specificity | —                            | 85.00%   |

> See [`docs/results.md`](docs/results.md) for the full per-epoch training history, convergence analysis, and comparison against baselines.

---

## Documentation

| Document                                         | Description                                                   |
| ------------------------------------------------ | ------------------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md)   | Model design, gating mechanism, CNN backbone, fusion head     |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | Dataset, preprocessing, augmentation, feature extraction, PCA |
| [`docs/training.md`](docs/training.md)           | Training loop, optimizer, LR scheduler, checkpointing         |
| [`docs/inference_api.md`](docs/inference_api.md) | REST API endpoints, request/response schema, ONNX runtime     |
| [`docs/configuration.md`](docs/configuration.md) | All hyperparameters and path constants reference              |
| [`docs/results.md`](docs/results.md)             | Training history, metrics, performance analysis               |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request

---

## Acknowledgements

- **Dataset:** [Parkinson's Drawings — Kaggle (kmader)](https://www.kaggle.com/datasets/kmader/parkinsons-drawings)
- **Backbone:** [EfficientNet-B0 via timm](https://github.com/huggingface/pytorch-image-models)
- **Feature Extraction:** [scikit-image HOG & LBP](https://scikit-image.org/)
