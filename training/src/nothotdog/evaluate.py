"""Held-out evaluation, with a per-class breakdown of which lookalikes still trip the model."""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf

from .config import ARTIFACT_DIR, ensure_dirs
from .dataset import GROUP_NAMES, food101_labels
from .embed import SCREENED
from .train import load_cache

SCREEN_GROUP_NAME = "hotdog_on_screen"


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    return {
        "truePositive": int(((predictions == 1) & (labels == 1)).sum()),
        "falsePositive": int(((predictions == 1) & (labels == 0)).sum()),
        "trueNegative": int(((predictions == 0) & (labels == 0)).sum()),
        "falseNegative": int(((predictions == 0) & (labels == 1)).sum()),
    }


def _rates(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = cm["truePositive"], cm["falsePositive"], cm["falseNegative"]
    total = sum(cm.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + cm["trueNegative"]) / total if total else 0.0,
        "hotdogPrecision": precision,
        "hotdogRecall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def evaluate(scores: np.ndarray, data: dict[str, np.ndarray], threshold: float) -> dict:
    labels = data["labels"].astype(int)
    predictions = (scores >= threshold).astype(int)

    names = food101_labels()
    per_class: dict[str, dict] = {}
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        group_name = SCREEN_GROUP_NAME if group == SCREENED else GROUP_NAMES[int(group)]
        per_class[group_name] = {
            "count": int(mask.sum()),
            "meanScore": float(scores[mask].mean()),
            "calledHotdog": int(predictions[mask].sum()),
        }

    hard_mask = data["group"] == 1
    hard_breakdown = {}
    for label in np.unique(data["food101Label"][hard_mask]):
        mask = hard_mask & (data["food101Label"] == label)
        hard_breakdown[names[int(label)]] = {
            "count": int(mask.sum()),
            "falsePositives": int(predictions[mask].sum()),
            "meanScore": float(scores[mask].mean()),
        }

    cm = _confusion(labels, predictions)
    return {
        "threshold": threshold,
        "confusion": cm,
        **_rates(cm),
        "byGroup": per_class,
        "hardNegatives": dict(
            sorted(hard_breakdown.items(), key=lambda kv: -kv[1]["falsePositives"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ARTIFACT_DIR / "not-hotdog.keras"))
    args = parser.parse_args()

    ensure_dirs()
    threshold = json.loads((ARTIFACT_DIR / "threshold.json").read_text())["threshold"]
    data = load_cache("validation")

    model = tf.keras.models.load_model(args.model)
    head = model.get_layer("head") if any(l.name == "head" for l in model.layers) else model
    scores = head.predict(data["embeddings"], verbose=0).ravel()

    report = evaluate(scores, data, threshold)
    path = ARTIFACT_DIR / "evaluation.json"
    path.write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k != "hardNegatives"}, indent=2))
    print("\nworst lookalikes:")
    for name, stats in list(report["hardNegatives"].items())[:8]:
        rate = stats["falsePositives"] / stats["count"] if stats["count"] else 0
        print(f"  {name:28s} {stats['falsePositives']:4d}/{stats['count']:<5d} ({rate:.1%})")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
