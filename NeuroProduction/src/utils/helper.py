import os
import joblib
from fastapi import FastAPI
from typing import AsyncGenerator
from src.constant.constant import *
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from src.inference.neurovox_predictor import NeuroVoxPredictor
from src.inference.neurospiral_predictor import NeuroSpiralPredictor

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    os.makedirs(TMP_DIR, exist_ok=True)
    app.state.executor = ThreadPoolExecutor(max_workers=4)

    print("Loading models…")
    app.state.voice_model = NeuroVoxPredictor(voice_model_path)

    reducers = joblib.load(reducers_path)
    app.state.spiral_model = NeuroSpiralPredictor(
        model_path=spiral_model_path,
        variance_selector=reducers["variance_selector"],
        scaler=reducers["scaler"],
        pca=reducers["pca"],
    )
    print("Models loaded.")

    yield

    app.state.executor.shutdown(wait=True)
    print("Executor shut down.")


def remove_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error deleting file {path}: {e}")
