from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .inference import InvalidImageError, encode_bytes, encode_png, predict, predict_onnx, prepare_image
from .model import load_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "weights" / "best_fold_3.weights.h5"
DEFAULT_ONNX = ROOT / "weights" / "nuraformer_fp32.onnx"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
model: tf.keras.Model | None = None
model_error: str | None = None
onnx_session = None
model_initialization_ms: float | None = None
execution_device: str | None = None
prediction_store: dict[str, tuple[str, bytes, bytes]] = {}
logger = logging.getLogger("nuraformer")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_error, onnx_session, model_initialization_ms, execution_device
    started = time.perf_counter()
    weights_path = Path(os.getenv("NURAFORMER_WEIGHTS", str(DEFAULT_WEIGHTS)))
    try:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        onnx_path = Path(os.getenv("NURAFORMER_ONNX", str(DEFAULT_ONNX)))
        if onnx_path.is_file():
            import onnxruntime as ort
            onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            model = None
            execution_device = "CPUExecutionProvider"
            logger.info("Using ONNX Runtime model from %s", onnx_path)
        else:
            model = load_model(str(weights_path))
            execution_device = "TensorFlow runtime"
        logger.info("Warming up NuRAFormer inference graph")
        if model is not None:
            model(np.zeros((1, 512, 512, 3), dtype=np.float32), training=False)
        model_error = None
        model_initialization_ms = (time.perf_counter() - started) * 1000
        app.state.model = model
        logger.info("NuRAFormer model initialized in %.3f ms", model_initialization_ms)
    except Exception as exc:
        model = None; app.state.model = None; model_error = str(exc)
        model_initialization_ms = (time.perf_counter() - started) * 1000
        logger.exception("NuRAFormer model failed to load")
    yield
    model = None; onnx_session = None; app.state.model = None


app = FastAPI(title="NuRAFormer Inference API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("NURAFORMER_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok" if model is not None or onnx_session is not None else "error", "model_loaded": model is not None or onnx_session is not None, "model_error": model_error, "model_initialization_ms": model_initialization_ms, "execution_device": execution_device}


@app.post("/predict")
def prediction(file: UploadFile = File(...)) -> dict[str, object]:
    request_started = time.perf_counter()
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Only PNG and JPG/JPEG images are supported.")
    if model is None and onnx_session is None:
        raise HTTPException(status_code=503, detail=f"NuRAFormer model is not available: {model_error}")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large.")
    try:
        preprocessing_started = time.perf_counter()
        prepared = prepare_image(data, file.filename)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000
        inference_started = time.perf_counter()
        if onnx_session is not None:
            segmentation, boundary = predict_onnx(onnx_session, prepared)
        else:
            segmentation, boundary = predict(model, prepared)
        # Both inference functions are synchronous: the call returns after the
        # forward pass completes, so no artificial device synchronization is needed.
        inference_ms = (time.perf_counter() - inference_started) * 1000
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    postprocessing_started = time.perf_counter()
    segmentation_png = encode_png(segmentation)
    boundary_png = encode_png(boundary)
    segmentation_probability = encode_png(segmentation, threshold=False)
    boundary_probability = encode_png(boundary, threshold=False)
    prediction_id = uuid.uuid4().hex
    stem = Path(file.filename or "sample").stem
    prediction_store[prediction_id] = (stem, base64.b64decode(segmentation_png), base64.b64decode(boundary_png))
    postprocessing_ms = (time.perf_counter() - postprocessing_started) * 1000
    return {"prediction_id": prediction_id, "width": 512, "height": 512, "threshold": 0.5, "original_image": f"data:image/png;base64,{encode_bytes(prepared.original_png)}", "segmentation_mask": f"data:image/png;base64,{segmentation_png}", "boundary_mask": f"data:image/png;base64,{boundary_png}", "segmentation_probability": f"data:image/png;base64,{segmentation_probability}", "boundary_probability": f"data:image/png;base64,{boundary_probability}", "timings": {"preprocessing_ms": round(preprocessing_ms, 3), "inference_ms": round(inference_ms, 3), "postprocessing_ms": round(postprocessing_ms, 3), "total_request_ms": round((time.perf_counter() - request_started) * 1000, 3)}}


@app.get("/predictions/{prediction_id}/masks.zip")
def download_masks(prediction_id: str) -> StreamingResponse:
    stored = prediction_store.get(prediction_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Prediction has expired or does not exist.")
    stem, segmentation_png, boundary_png = stored
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{stem}_segmentation.png", segmentation_png)
        bundle.writestr(f"{stem}_boundary.png", boundary_png)
    archive.seek(0)
    return StreamingResponse(archive, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{stem}_masks.zip"'})


@app.get("/predictions/batch.zip")
def download_batch(prediction_ids: str) -> StreamingResponse:
    ids = [item for item in prediction_ids.split(",") if item]
    if not ids:
        raise HTTPException(status_code=400, detail="No completed predictions were supplied.")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for prediction_id in ids:
            stored = prediction_store.get(prediction_id)
            if stored is None:
                continue
            stem, segmentation_png, boundary_png = stored
            bundle.writestr(f"{stem}/segmentation_mask.png", segmentation_png)
            bundle.writestr(f"{stem}/boundary_mask.png", boundary_png)
    archive.seek(0)
    return StreamingResponse(archive, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="NuRAFormer_Batch_Results.zip"'})
