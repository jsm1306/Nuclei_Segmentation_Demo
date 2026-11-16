from __future__ import annotations

import io
import base64
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


def _sample_microscopy_image() -> bytes:
    image_path = next((Path(__file__).parents[1] / "Images").glob("*.png"))
    return image_path.read_bytes()


def test_predict_returns_both_nonempty_masks_and_reuses_loaded_model() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        payload = {"file": ("sample.png", _sample_microscopy_image(), "image/png")}
        loaded_model_id = id(app.state.model) if hasattr(app.state, "model") else None
        first = client.post("/predict", files=payload)

        assert first.status_code == 200
        body = first.json()
        assert body["width"] == 512
        assert body["height"] == 512
        processed = body["original_image"].split(",", 1)[1]
        processed_image = np.asarray(
            Image.open(io.BytesIO(base64.b64decode(processed)))
        )
        assert processed_image.shape == (512, 512, 3)
        for key in ("segmentation_mask", "boundary_mask"):
            assert body[key].startswith("data:image/png;base64,")
            encoded = body[key].split(",", 1)[1]
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))))
            assert mask.shape == (512, 512)
            assert np.count_nonzero(mask) > 0
        assert loaded_model_id == id(app.state.model)
        assert client.get("/health").json()["model_loaded"] is True
