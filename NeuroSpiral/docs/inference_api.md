# Inference & API

This document covers the ONNX inference engine, the `NeuroSpiralPredictor` class, the FastAPI REST server, all available endpoints, request/response formats, and error handling.

---

## ONNX Inference

### Why ONNX?

Exporting to ONNX decouples the deployment environment from the PyTorch training stack. The production server requires only `onnxruntime` — a lightweight C++ runtime with Python bindings — with no dependency on PyTorch, timm, or GPU drivers.

| Benefit     | Detail                                                 |
| ----------- | ------------------------------------------------------ |
| Portability | Runs on CPU, GPU, or edge devices without PyTorch      |
| Performance | ONNX Runtime applies graph optimizations automatically |
| Stability   | Fixed computation graph — no Python overhead per call  |

---

### Model Inputs & Outputs

| Name               | Shape              | Dtype     | Description                                                        |
| ------------------ | ------------------ | --------- | ------------------------------------------------------------------ |
| `image`            | `[B, 3, 224, 224]` | `float32` | Normalised grayscale image replicated to 3 channels, ∈ [0, 1]      |
| `math_features`    | `[B, K]`           | `float32` | PCA-reduced HOG + LBP feature vector (K determined at export time) |
| `logit` _(output)_ | `[B, 1]`           | `float32` | Raw logit — apply sigmoid to obtain probability                    |

The batch dimension `B` is **dynamic** — the same ONNX model handles any batch size without re-export.

> **Note on `K`:** The `math_features` dimension reflects the PCA output from the training run. Export the model immediately after training to ensure the ONNX graph matches the reducer pipeline.

---

## NeuroSpiralPredictor

**Class:** `NeuroSpiralPredictor`  
**Location:** `src/inference/predictor.py`

Wraps the ONNX Runtime session and exposes a single `predict()` method. Handles the full preprocessing → feature extraction → inference → postprocessing pipeline.

### Initialisation

```python
predictor = NeuroSpiralPredictor(model_path="spiral_best_model.onnx")
```

On startup the class:

1. Creates an `ort.InferenceSession` with the configured execution providers
2. Caches input names and output name from the session metadata

---

### `predict(img)` — Full Pipeline

**Input:** Raw BGR `np.ndarray`, any resolution (resized internally)

| Step | Operation                                                    |
| ---- | ------------------------------------------------------------ |
| 1    | Resize image to 224×224                                      |
| 2    | Apply preprocessing: Fourier LPF → Otsu → Morphology         |
| 3    | Build image tensor `(1, 3, 224, 224)`, normalise to [0, 1]   |
| 4    | Extract HOG + LBP → apply VT + Scaler + PCA → shape `(1, K)` |
| 5    | Run ONNX session with both inputs                            |
| 6    | Apply sigmoid to raw logit                                   |
| 7    | Apply threshold 0.46: `< 0.46` → `"PD"`, `≥ 0.46` → `"HC"`   |

> **Threshold:** The decision threshold is **0.46**, not 0.5. This value was selected by maximizing F1 on the validation set after training and is stored as a constant alongside the ONNX model.

**Output:** `Tuple[str, float]`

```python
label, probability = predictor.predict(img)
# label       → "HC" or "PD"
# probability → float ∈ [0.0, 1.0], rounded to 4 decimal places
```

---

### Example

```python
import cv2
from src.inference.predictor import NeuroSpiralPredictor

predictor = NeuroSpiralPredictor("spiral_best_model.onnx")

img = cv2.imread("test_spiral.png")
label, prob = predictor.predict(img)

print(f"Prediction: {label} (probability={prob:.4f})")
# Prediction: HC (probability=0.8342)
```

---

## REST API

| Property         | Value                                     |
| ---------------- | ----------------------------------------- |
| Framework        | FastAPI                                   |
| Server           | Uvicorn (ASGI)                            |
| Base URL         | `http://localhost:8000`                   |
| Interactive docs | `http://localhost:8000/docs` (Swagger UI) |

---

### Application Lifecycle

The API uses FastAPI's `lifespan` context manager to load the ONNX predictor **once at startup** and release it cleanly on shutdown:

```python
@asynccontextmanager
async def lifespan(app):
    ml_models["predictor"] = NeuroSpiralPredictor(ONNX_EXPORT_PATH)
    yield
    ml_models.clear()
```

The predictor (including the ONNX session and all reducer objects) is loaded exactly once and shared across all requests — eliminating per-request model loading overhead.

---

## Endpoints

### `GET /health`

Lightweight health check to verify the server and model are ready.

**Response — 200 OK**

```json
{
  "status": "ok",
  "model": "NeuroSpiral"
}
```

---

### `POST /predict`

Classify an uploaded drawing as Healthy Control (`HC`) or Parkinson's Disease (`PD`).

**Request**

| Field   | Type                         | Required | Description                                              |
| ------- | ---------------------------- | -------- | -------------------------------------------------------- |
| `image` | `file` (multipart/form-data) | ✅       | Drawing image (PNG, JPG, or any OpenCV-decodable format) |

**Response — 200 OK**

```json
{
  "label": "HC",
  "probability": 0.8342
}
```

| Field         | Type     | Description                                              |
| ------------- | -------- | -------------------------------------------------------- |
| `label`       | `string` | `"HC"` (Healthy Control) or `"PD"` (Parkinson's Disease) |
| `probability` | `float`  | Sigmoid-transformed model confidence ∈ [0.0, 1.0]        |

**Error Responses**

| Code                         | Condition                                                   |
| ---------------------------- | ----------------------------------------------------------- |
| `415 Unsupported Media Type` | Uploaded file's `content_type` does not start with `image/` |
| `400 Bad Request`            | Image cannot be decoded (corrupted or invalid file content) |
| `500 Internal Server Error`  | Unhandled exception during preprocessing or ONNX inference  |

---

## Async Inference

ONNX inference is CPU-bound and would block FastAPI's async event loop if called directly. The endpoint offloads it to a thread pool:

```python
label, probability = await asyncio.to_thread(predictor.predict, img)
```

This keeps the event loop free to accept and route new requests concurrently while inference executes in a background thread.

---

## Running the Server

### Development (single worker, auto-reload)

```bash
python src/main.py
```

### Production (multiple workers)

```bash
uvicorn src.api.endpoint:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Memory note:** Each worker process loads its own copy of the ONNX model and reducer objects. With 4 workers, expect ~4× the per-process memory footprint (~500MB–1GB total).

---

## Client Examples

### Python (`requests`)

```python
import requests

with open("spiral.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"image": ("spiral.png", f, "image/png")},
    )

result = response.json()
print(result)
# {"label": "PD", "probability": 0.2731}
```

### curl

```bash
curl -X POST http://localhost:8000/predict \
     -F "image=@spiral.png" \
     -H "accept: application/json"
```

### JavaScript (Fetch API)

```javascript
const formData = new FormData();
formData.append("image", fileInput.files[0]);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result);
// { label: "HC", probability: 0.8342 }
```

---

## Response Schema (Pydantic)

```python
class PredictionResponse(BaseModel):
    label: str         # "HC" or "PD"
    probability: float # ∈ [0.0, 1.0]
```

Pydantic validates all output types before serialisation. A malformed model output (e.g., NaN logit) would raise a `ValidationError`, which FastAPI catches and returns as a 500 response.

---

## Preprocessing Consistency

The predictor applies **exactly the same preprocessing pipeline** used during training:

| Stage                  | Training                         | Inference                              |
| ---------------------- | -------------------------------- | -------------------------------------- |
| Resize                 | ✅ 224×224                       | ✅ 224×224                             |
| Fourier LPF            | ✅ cutoff=30                     | ✅ cutoff=30                           |
| Otsu binarization      | ✅                               | ✅                                     |
| Morphological cleaning | ✅ kernel=3, iter=1              | ✅ kernel=3, iter=1                    |
| Feature extraction     | HOG + LBP                        | HOG + LBP                              |
| Feature reduction      | VT + Scaler + PCA (fit on train) | Same fitted objects (loaded from disk) |
| Decision threshold     | 0.46 (tuned on val)              | 0.46                                   |

Any deviation between training and inference preprocessing would silently degrade performance. The predictor loads the fitted reducer objects (`vt`, `scaler`, `pca`) from the same checkpoint directory as the ONNX model.
