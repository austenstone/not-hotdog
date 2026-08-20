"""Two-stage training.

Stage A trains a small head on cached frozen-base embeddings. It takes seconds and is where
threshold selection happens.

Stage B unfreezes the top of MobileNetV2 and fine-tunes on real images. It is the only slow step
and is optional -- if it is too painful on CPU, the Stage A model is already shippable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve

from .augment import to_model_input
from .config import (
    ARTIFACT_DIR,
    EASY_NEGATIVE_TAKE,
    EMBEDDING_DIM,
    IMAGE_SHAPE,
    NONFOOD_TAKE,
    SEED,
    TARGET_FPR,
    ensure_dirs,
)
from .embed import base_model, cache_path, is_hotdog, prepare

AUTOTUNE = tf.data.AUTOTUNE
FINETUNE_LAYERS = 30


def build_head() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(EMBEDDING_DIM,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="head",
    )


def load_cache(split: str) -> dict[str, np.ndarray]:
    with np.load(cache_path(split)) as data:
        return {k: data[k] for k in data.files}


def select_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pick the operating point by false-positive rate, not precision.

    A false "HOTDOG" is the funnier and more damaging failure, so we still buy correctness with
    recall. But precision is a function of class prevalence, and the prevalence in this validation
    set is an artefact of how the negatives were curated (~8 negatives per hotdog) rather than
    anything a user experiences. Someone pointing a camera at their lunch has a completely
    different prior. Tuning to a precision target therefore optimises against a number that does
    not transfer.

    False-positive rate is prevalence-independent: "how often does a non-hotdog trigger the
    overlay" means the same thing here and in the wild. So we cap FPR and take the best recall
    available under that cap.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    viable = np.where(fpr <= TARGET_FPR)[0]
    best = viable[int(np.argmax(tpr[viable]))]
    return float(thresholds[best])


def train_head(epochs: int) -> tuple[tf.keras.Model, float, dict]:
    train = load_cache("train")
    validation = load_cache("validation")

    labels = train["labels"]
    positives = float(labels.sum())
    total = float(len(labels))
    class_weight = {
        0: total / (2.0 * (total - positives)),
        1: total / (2.0 * positives),
    }

    head = build_head()
    head.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    history = head.fit(
        train["embeddings"],
        labels,
        validation_data=(validation["embeddings"], validation["labels"]),
        epochs=epochs,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=6, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    scores = head.predict(validation["embeddings"], verbose=0).ravel()
    threshold = select_threshold(validation["labels"], scores)
    return head, threshold, {k: float(v[-1]) for k, v in history.history.items()}


def assemble(head: tf.keras.Model) -> tf.keras.Model:
    base = base_model()
    inputs = tf.keras.Input(shape=IMAGE_SHAPE, name="image")
    outputs = head(base(inputs, training=False))
    return tf.keras.Model(inputs, outputs, name="not_hotdog")


def image_pipeline(split: str, batch: int, shuffle: bool) -> tf.data.Dataset:
    easy = EASY_NEGATIVE_TAKE if split == "train" else EASY_NEGATIVE_TAKE // 4
    nonfood = NONFOOD_TAKE if split == "train" else NONFOOD_TAKE // 4
    ds = prepare(split, easy, nonfood, interleave=shuffle)
    ds = ds.map(
        lambda image, label, group: (
            to_model_input(image),
            tf.cast(is_hotdog(group), tf.float32),
        ),
        num_parallel_calls=AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(4096, seed=SEED)
    return ds.batch(batch).prefetch(AUTOTUNE)


def finetune(
    model: tf.keras.Model, epochs: int, batch: int, steps: int, validation_steps: int
) -> tf.keras.Model:
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-FINETUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.fit(
        image_pipeline("train", batch, shuffle=True).repeat(),
        validation_data=image_pipeline("validation", batch, shuffle=False).take(validation_steps),
        epochs=epochs,
        steps_per_epoch=steps,
        verbose=1,
    )
    return model


def score_images(model: tf.keras.Model) -> dict[str, np.ndarray]:
    """Run the full model over the validation images.

    Scoring through the cached embeddings would be much faster, but the cache is only valid while
    the base network is frozen. After Stage B the base has moved, so cached vectors describe a
    model that no longer exists — and the evaluation would silently measure the wrong thing.
    """
    ds = prepare("validation", EASY_NEGATIVE_TAKE // 4, NONFOOD_TAKE // 4)
    ds = ds.map(lambda i, l, g: (to_model_input(i), l, g), num_parallel_calls=tf.data.AUTOTUNE)

    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for images, label, group in ds.batch(64).prefetch(tf.data.AUTOTUNE):
        scores.append(model(images, training=False).numpy().ravel())
        labels.append(label.numpy())
        groups.append(group.numpy())

    group_array = np.concatenate(groups)
    return {
        "scores": np.concatenate(scores),
        "labels": is_hotdog(tf.constant(group_array)).numpy().astype(np.float32),
        "food101Label": np.concatenate(labels),
        "group": group_array,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--finetune-batch", type=int, default=32)
    # Fine-tuning is the only genuinely slow step (no GPU), so it is bounded by step count rather
    # than by full passes over the 48k training images.
    parser.add_argument("--finetune-steps", type=int, default=300)
    parser.add_argument("--finetune-validation-steps", type=int, default=60)
    args = parser.parse_args()

    ensure_dirs()
    head, threshold, metrics = train_head(args.head_epochs)
    model = assemble(head)

    if args.finetune_epochs > 0:
        model = finetune(
            model,
            args.finetune_epochs,
            args.finetune_batch,
            args.finetune_steps,
            args.finetune_validation_steps,
        )
        # Scores and labels must come from the same pass. Pairing fresh predictions against the
        # cached labels relied on two pipelines emitting identical order, which is not a
        # guarantee worth betting the operating point on.
        scored = score_images(model)
        threshold = select_threshold(scored["labels"], scored["scores"])
        metrics = {"stage": "finetune", "validationCount": int(len(scored["scores"]))}

    path = ARTIFACT_DIR / "not-hotdog.keras"
    model.save(path)
    (ARTIFACT_DIR / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "targetFpr": TARGET_FPR, **metrics}, indent=2)
    )
    print(f"threshold {threshold:.4f} -> {path}")


if __name__ == "__main__":
    main()
