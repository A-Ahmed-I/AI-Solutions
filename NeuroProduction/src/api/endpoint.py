import cv2
import uuid
import asyncio
import numpy as np
from typing import Any
from src.utils.helper import *
from src.api.schema import PredictionResponse
from fastapi import UploadFile, File, HTTPException, status, BackgroundTasks

app = FastAPI(lifespan=lifespan, title="NeuroVive API")


async def _run_in_executor(app: FastAPI, fn, *args) -> Any:
    """
    Executes a CPU-bound function in a separate thread to avoid blocking the event loop.

    Args:
        app (FastAPI): The FastAPI application instance.
        fn (callable): The function to execute.
        *args: Variable arguments to pass to the function.

    Returns:
        any: The result of the function execution.
    """
    loop = asyncio.get_running_loop()

    return await loop.run_in_executor(app.state.executor, fn, *args)


@app.post("/predict/image", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_image(image: UploadFile = File(...)) -> PredictionResponse:
    """
    Endpoint to receive an image file, process it, and return a model prediction.

    This function validates the uploaded file type, decodes the image into a
    NumPy array, and delegates the CPU-bound prediction task to a separate
    thread to avoid blocking the asynchronous event loop.

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

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded file must be an image (JPEG, PNG, …).",
        )

    content: bytes = await image.read()
    np_arr: np.ndarray = np.frombuffer(content, np.uint8)
    img: np.ndarray = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. The file may be corrupt or unsupported.",
        )

    try:
        label, probability = await _run_in_executor(app, app.state.spiral_model.predict, img)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        ) from exc

    return PredictionResponse(label=label, probability=round(float(probability), 4))


@app.post("/predict/voice", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_audio(background_task: BackgroundTasks, audio: UploadFile = File(...)) -> PredictionResponse:
    """
    Endpoint to receive a WAV audio file, save it temporarily, process it
    via the voice model, and return a prediction.

    Args:
        background_task (BackgroundTasks): FastAPI utility to clean up temporary files.
        audio (UploadFile): The .wav audio file uploaded via form data.

    Returns:
        PredictionResponse: A Pydantic model containing the prediction `label`
                            (str) and `probability` (float).

    Raises:
        HTTPException:
            - 415: If the file is not a .wav file.
            - 500: If processing fails or an error occurs during inference.
    """

    if not audio.filename or not audio.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .wav audio files are supported.",
        )

    tmp_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}.wav")

    try:
        with open(tmp_path, "wb") as buffer:
            while content := await audio.read(1024 * 1024):
                buffer.write(content)

        label, probability = await _run_in_executor(
            app, app.state.voice_model.predict, tmp_path
        )

        if label == "Error":
            remove_file(tmp_path)
            raise HTTPException(status_code=500, detail="Processing failed")

        background_task.add_task(remove_file, tmp_path)

        return PredictionResponse(label=label, probability=round(float(probability), 4))

    except Exception as e:
        remove_file(tmp_path)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
