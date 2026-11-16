from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (512, 512)
ALLOWED_FORMATS = {"PNG", "JPEG"}
THRESHOLD = 0.5


class InvalidImageError(ValueError):
    """Raised when an upload is not a supported, decodable image."""


@dataclass(frozen=True)
class PreparedImage:
    model_input: np.ndarray
    original_png: bytes


def prepare_image(data: bytes, filename: str | None = None) -> PreparedImage:
    if not data:
        raise InvalidImageError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise InvalidImageError("Only PNG and JPG/JPEG images are supported.")
            rgb_image = image.convert("RGB")
            resized = rgb_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            processed_buffer = io.BytesIO()
            resized.save(processed_buffer, format="PNG")
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise InvalidImageError("The uploaded image could not be read.") from exc

    # The notebook used cv2.imread (BGR) followed by /255.0. Decode as RGB
    # for the API, then restore that training-time channel order for inference.
    bgr = np.asarray(resized, dtype=np.float32)[:, :, ::-1] / 255.0
    return PreparedImage(
        model_input=np.expand_dims(bgr, axis=0),
        original_png=processed_buffer.getvalue(),
    )


def _validate_output(name: str, output: object) -> np.ndarray:
    array = np.asarray(output)
    if array.shape != (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1):
        raise RuntimeError(
            f"Unexpected {name} output shape {array.shape}; "
            f"expected (1, {IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}, 1)."
        )
    return np.clip(array[0, :, :, 0], 0.0, 1.0)


def predict(model: tf.keras.Model, prepared: PreparedImage) -> tuple[np.ndarray, np.ndarray]:
    outputs = model(prepared.model_input, training=False)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise RuntimeError("The model must return segmentation and boundary outputs.")
    segmentation = _validate_output("segmentation", outputs[0])
    boundary = _validate_output("boundary", outputs[1])
    return segmentation, boundary


def predict_onnx(session: object, prepared: PreparedImage) -> tuple[np.ndarray, np.ndarray]:
    """Run the exported model through ONNX Runtime."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: prepared.model_input})
    if len(outputs) != 2:
        raise RuntimeError("The ONNX model must return segmentation and boundary outputs.")
    return _validate_output("segmentation", outputs[0]), _validate_output("boundary", outputs[1])


def encode_png(mask: np.ndarray, threshold: bool = True) -> str:
    values = (mask > THRESHOLD).astype(np.uint8) * 255 if threshold else (
        np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    )
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
