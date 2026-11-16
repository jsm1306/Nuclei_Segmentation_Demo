from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    InvalidImageError, encode_bytes, encode_png, predict, predict_onnx, prepare_image,
)
from .model import load_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "best_fold_3.weights.h5"
DEFAULT_ONNX = ROOT / "weights" / "nuraformer_fp32.onnx"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
model: tf.keras.Model | None = None
model_error: str | None = None
onnx_session = None
logger = logging.getLogger("nuraformer")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_error, onnx_session
    weights_path = Path(os.getenv("NURAFORMER_WEIGHTS", str(DEFAULT_WEIGHTS)))
    try:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        onnx_path = Path(os.getenv("NURAFORMER_ONNX", str(DEFAULT_ONNX)))
        if onnx_path.is_file():
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            model = None
            logger.info("Using ONNX Runtime model from %s", onnx_path)
        else:
            model = load_model(str(weights_path))
        # Trace the model before accepting requests.  On CPU the first Keras
        # call includes graph construction, which otherwise makes the browser
        # look stuck in “Running inference”.
        logger.info("Warming up NuRAFormer inference graph")
        if model is not None:
            model(np.zeros((1, 512, 512, 3), dtype=np.float32), training=False)
        model_error = None
        app.state.model = model
        logger.info("NuRAFormer model loaded from %s", weights_path)
    except Exception as exc:
        model = None
        app.state.model = None
        model_error = str(exc)
        logger.exception("NuRAFormer model failed to load")
    yield
    model = None
    onnx_session = None
    app.state.model = None


app = FastAPI(title="NuRAFormer Inference API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "NURAFORMER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if model is not None or onnx_session is not None else "error",
        "model_loaded": model is not None or onnx_session is not None,
        "model_error": model_error,
    }


@app.post("/predict")
def prediction(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPG/JPEG images are supported.",
        )
    if model is None and onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail=f"NuRAFormer model is not available: {model_error}",
        )
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")
    try:
        logger.info("Starting inference for %s (%d bytes)", file.filename, len(data))
        prepared = prepare_image(data, file.filename)
        if onnx_session is not None:
            segmentation, boundary = predict_onnx(onnx_session, prepared)
        else:
            segmentation, boundary = predict(model, prepared)
        logger.info("Inference completed for %s", file.filename)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "width": 512,
        "height": 512,
        "threshold": 0.5,
        "original_image": f"data:image/png;base64,{encode_bytes(prepared.original_png)}",
        "segmentation_mask": f"data:image/png;base64,{encode_png(segmentation)}",
        "boundary_mask": f"data:image/png;base64,{encode_png(boundary)}",
        "segmentation_probability": f"data:image/png;base64,{encode_png(segmentation, threshold=False)}",
        "boundary_probability": f"data:image/png;base64,{encode_png(boundary, threshold=False)}",
    }
