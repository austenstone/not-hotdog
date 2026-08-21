"""Score the committed parity fixture with the shipped .tflite.

The web and mobile clients feed the model 8-bit pixels decoded from an image, so
this reference does the same rather than scoring a float tensor straight out of
the training pipeline. Scoring the float tensor instead drifts by a couple of
points purely from the uint8 round-trip, which reads like a client bug when it
is really a fixture artifact.

    uv run --project training python scripts/parity.py
    uv run --project training python scripts/parity.py --write

Compare the printed scores against what a client renders for the same files.
"""

import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "parity"
EXPECTED = FIXTURE / "expected.json"


def score(interpreter, pixels, normalization):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    values = pixels * normalization["scale"] + normalization["offset"]
    if in_scale:
        values = np.round(values / in_scale + in_zero)

    interpreter.set_tensor(inp["index"], values[None, ...].astype(inp["dtype"]))
    interpreter.invoke()
    raw = float(interpreter.get_tensor(out["index"])[0][0])
    return (raw - out_zero) * out_scale if out_scale else raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="rewrite expected.json")
    args = parser.parse_args()

    metadata = json.loads((ROOT / "models" / "metadata.json").read_text())
    threshold = metadata["threshold"]

    interpreter = tf.lite.Interpreter(model_path=str(ROOT / "models" / "not-hotdog.tflite"))
    interpreter.allocate_tensors()

    results = {}
    for image in sorted(FIXTURE.glob("*.png")):
        pixels = tf.io.decode_png(tf.io.read_file(str(image)), channels=3)
        results[image.name] = round(score(interpreter, pixels.numpy().astype(np.float32), metadata["normalization"]), 6)

    previous = json.loads(EXPECTED.read_text()) if EXPECTED.exists() else {}
    width = max(len(name) for name in results)
    drifted = False

    for name, value in results.items():
        verdict = "HOTDOG" if value >= threshold else "NOT HOTDOG"
        line = f"{name:<{width}}  {value * 100:6.2f}%  {verdict}"
        if name in previous.get("scores", {}):
            delta = value - previous["scores"][name]
            if abs(delta) > 1e-6:
                drifted = True
                line += f"  (was {previous['scores'][name] * 100:.2f}%)"
        print(line)

    print(f"\nthreshold {threshold * 100:.2f}%  model {metadata.get('gitSha', 'unknown')}")

    if args.write:
        EXPECTED.write_text(json.dumps({"threshold": threshold, "scores": results}, indent=2) + "\n")
        print(f"wrote {EXPECTED.relative_to(ROOT)}")
    elif drifted:
        print("\nScores moved. Re-run with --write once you have confirmed the new model is the one you want.")


if __name__ == "__main__":
    main()
