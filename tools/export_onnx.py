"""Export NuRAFormer to ONNX and create a CPU-friendly INT8 model."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from backend.model import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/best_fold_3.weights.h5")
    parser.add_argument("--output", default="weights/nuraformer_fp32.onnx")
    parser.add_argument("--quantized-output", default="weights/nuraformer_int8.onnx")
    args = parser.parse_args()

    model = load_model(args.weights)
    input_signature = [tf.TensorSpec((1, 512, 512, 3), tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=17,
        output_path=args.output,
    )
    print(f"Exported {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")
    quantize_dynamic(args.output, args.quantized_output, weight_type=QuantType.QInt8)
    print(f"Exported {args.quantized_output} ({Path(args.quantized_output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
