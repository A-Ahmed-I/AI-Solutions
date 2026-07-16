import cv2
import joblib
import asyncio
import numpy as np
from src.constant.constant import *
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
from src.api.schema import PredictionResponse
from src.inference.predictor import NeuroSpiralPredictor
from fastapi import FastAPI, UploadFile, File, HTTPException, status

ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager to handle the lifespan events of the FastAPI application.

    Loads the NeuroSpiralPredictor model and fitted feature reducers into
    memory when the application starts, and clears resources on shutdown.

    Args:
        app (FastAPI): The instance of the FastAPI application.

    Yields:
        None: Yields control back to the FastAPI application.
    """
    reducers = joblib.load(REDUCERS_PATH)

    ml_models["predictor"] = NeuroSpiralPredictor(
        model_path=ONNX_EXPORT_PATH,
        variance_selector=reducers["variance_selector"],
        scaler=reducers["scaler"],
        pca=reducers["pca"],
    )

    yield

    ml_models.clear()


app: FastAPI = FastAPI(lifespan=lifespan, title="NeuroSpiral API")


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...)) -> PredictionResponse:
    """
    Endpoint to receive an image file, process it, and return a model prediction.

    Args:
        image (UploadFile): The image file uploaded via form data.

    Returns:
        PredictionResponse: A Pydantic model containing the prediction `label`
                            (str) and `probability` (float).

    Raises:
        HTTPException:
            - 415: If the uploaded file is not an image.
            - 400: If the image is invalid or cannot be decoded.
            - 500: If an error occurs during model inference.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image.",
        )

    content: bytes = await image.read()
    np_arr: np.ndarray = np.frombuffer(content, np.uint8)
    img: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image file.",
        )

    try:
        label, probability = await asyncio.to_thread(
            ml_models["predictor"].predict, img
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    return PredictionResponse(label=label, probability=round(float(probability), 4))
