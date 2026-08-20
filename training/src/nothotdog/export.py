"""Export to a quantized .tflite plus the metadata contract both clients read.

The same artifact runs in the browser via LiteRT.js and on device via react-native-fast-tflite,
so the decision threshold and normalization constants live in metadata.json rather than being
duplicated (and drifting) across two codebases.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import tensorflow as tf

from .config import (
    ARTIFACT_DIR,
    CLASS_ORDER,
    IMAGE_SHAPE,
    MODELS_DIR,
    ensure_dirs,
)
from .train import image_pipeline

REPRESENTATIVE_SAMPLES = 200


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def representative_dataset():
    ds = image_pipeline("train", batch=1, shuffle=True).take(REPRESENTATIVE_SAMPLES)
    for images, _ in ds:
        yield [tf.cast(images, tf.float32)]


def convert(model: tf.keras.Model, quantize: bool) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    return converter.convert()


def measure(tflite_model: bytes, embeddings_split: str, threshold: float) -> dict:
    """Compare quantized predictions against the float model on real validation images."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    scores: list[float] = []
    labels: list[float] = []
    # Must sample with interleaving on. Reading the stream in its natural order takes the
    # positives block and nothing else, which scores a precision of 1.0 against a sample that
    # contains no negatives to be wrong about.
    for images, label in image_pipeline(embeddings_split, batch=1, shuffle=True).take(1500):
        value = images.numpy().astype(input_detail["dtype"])
        if input_detail["dtype"] in (np.int8, np.uint8):
            scale, zero = input_detail["quantization"]
            value = (images.numpy() / scale + zero).astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], value)
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail["index"]).astype(np.float32)
        if output_detail["dtype"] in (np.int8, np.uint8):
            scale, zero = output_detail["quantization"]
            out = (out - zero) * scale
        scores.append(float(out.ravel()[0]))
        labels.append(float(label.numpy().ravel()[0]))

    score_array, label_array = np.array(scores), np.array(labels)
    predictions = (score_array >= threshold).astype(int)
    truth = label_array.astype(int)
    tp = int(((predictions == 1) & (truth == 1)).sum())
    fp = int(((predictions == 1) & (truth == 0)).sum())
    fn = int(((predictions == 0) & (truth == 1)).sum())
    correct = int((predictions == truth).sum())
    return {
        "sampled": len(scores),
        "accuracy": correct / len(scores) if scores else 0.0,
        "hotdogPrecision": tp / (tp + fp) if tp + fp else 0.0,
        "hotdogRecall": tp / (tp + fn) if tp + fn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ARTIFACT_DIR / "not-hotdog.keras"))
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    threshold = json.loads((ARTIFACT_DIR / "threshold.json").read_text())["threshold"]
    model = tf.keras.models.load_model(args.model)

    float_model = convert(model, quantize=False)
    int8_model = convert(model, quantize=True)

    target = MODELS_DIR / "not-hotdog.tflite"
    target.write_bytes(int8_model)

    metrics = {} if args.skip_verify else measure(int8_model, "validation", threshold)

    metadata = {
        "model": target.name,
        "inputShape": [1, *IMAGE_SHAPE],
        "inputDtype": "float32",
        "normalization": {"mode": "mobilenet_v2", "scale": 2.0 / 255.0, "offset": -1.0},
        "classOrder": CLASS_ORDER,
        "threshold": threshold,
        "gitSha": _git_sha(),
        "sizeBytes": {"float32": len(float_model), "int8": len(int8_model)},
        "metrics": metrics,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    reduction = 1 - len(int8_model) / len(float_model)
    print(
        f"float32 {len(float_model) / 1e6:.2f} MB -> int8 {len(int8_model) / 1e6:.2f} MB "
        f"({reduction:.0%} smaller)"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
