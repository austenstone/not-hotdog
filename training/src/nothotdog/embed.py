"""Stage A: cache frozen MobileNetV2 embeddings so head training costs seconds, not hours.

There is no usable GPU on this machine (tensorflow-metal caps at cp312 and breaks under Keras 3),
so the base network is run over the data exactly once and the 1280-d vectors are kept. Every
hyperparameter sweep after that is a matter of seconds on CPU.

Every cached vector is of an *augmented* image, including negatives. If only positives were
augmented the head would happily learn "rotation fill artefact => hotdog" instead of learning
what a hotdog looks like.
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from .augment import augment, screen_effect, to_model_input
from .config import (
    CACHE_DIR,
    EASY_NEGATIVE_TAKE,
    EMBEDDING_DIM,
    IMAGE_SHAPE,
    NONFOOD_TAKE,
    POSITIVE_VARIANTS,
    SCREEN_LABEL,
    SCREEN_VARIANTS,
    SEED,
    ensure_dirs,
)
from .dataset import HARD_NEGATIVE, HOTDOG, group_datasets

AUTOTUNE = tf.data.AUTOTUNE
BATCH = 64

SCREENED = 4
SCREEN_NEGATIVE_EVERY = 7


def base_model() -> tf.keras.Model:
    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SHAPE,
        pooling="avg",
    )
    model.trainable = False
    return model


def _seed_for(index: tf.Tensor, variant: int) -> tf.Tensor:
    return tf.stack([tf.cast(index, tf.int32) + SEED, tf.constant(variant, tf.int32)])


def _expand_positive(index, image, label, group):
    frames = [augment(image, _seed_for(index, v)) for v in range(POSITIVE_VARIANTS)]
    screens = [
        screen_effect(augment(image, _seed_for(index, 100 + v)), _seed_for(index, 200 + v))
        for v in range(SCREEN_VARIANTS)
    ]
    images = tf.stack(frames + screens)
    groups = tf.concat(
        [
            tf.fill([POSITIVE_VARIANTS], group),
            tf.fill([SCREEN_VARIANTS], tf.constant(SCREENED, tf.int32)),
        ],
        axis=0,
    )
    n = POSITIVE_VARIANTS + SCREEN_VARIANTS
    return tf.data.Dataset.from_tensor_slices((images, tf.fill([n], label), groups))


def _expand_plain(index, image, label, group):
    image = augment(image, _seed_for(index, 0))
    return tf.data.Dataset.from_tensors((image, label, group))


def _expand_screened_negative(index, image, label, group):
    image = screen_effect(augment(image, _seed_for(index, 0)), _seed_for(index, 300))
    return tf.data.Dataset.from_tensors((image, label, group))


def _prepare(split: str, easy_take: int, nonfood_take: int) -> tf.data.Dataset:
    groups = group_datasets(split, easy_take=easy_take, nonfood_take=nonfood_take)

    positives = groups[HOTDOG].enumerate().flat_map(
        lambda i, x: _expand_positive(i, x[0], x[1], x[2])
    )

    # Screen artefacts are sprinkled onto hard negatives too. Without this the head would learn
    # that moire means hotdog rather than learning what a screen looks like.
    hard = groups[HARD_NEGATIVE].enumerate()
    hard_screened = hard.filter(lambda i, x: i % SCREEN_NEGATIVE_EVERY == 0).flat_map(
        lambda i, x: _expand_screened_negative(i, x[0], x[1], x[2])
    )
    hard_plain = hard.filter(lambda i, x: i % SCREEN_NEGATIVE_EVERY != 0).flat_map(
        lambda i, x: _expand_plain(i, x[0], x[1], x[2])
    )

    combined = positives.concatenate(hard_plain).concatenate(hard_screened)
    for key, ds in groups.items():
        if key in (HOTDOG, HARD_NEGATIVE):
            continue
        combined = combined.concatenate(
            ds.enumerate().flat_map(lambda i, x: _expand_plain(i, x[0], x[1], x[2]))
        )

    return combined


def embed_split(split: str, easy_take: int, nonfood_take: int) -> dict[str, np.ndarray]:
    ds = _prepare(split, easy_take, nonfood_take)
    ds = ds.map(lambda i, l, g: (to_model_input(i), l, g), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)

    base = base_model()
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    seen = 0
    for images, label, group in ds:
        vectors.append(base(images, training=False).numpy().astype(np.float32))
        labels.append(label.numpy())
        groups.append(group.numpy())
        seen += int(images.shape[0])
        if seen % (BATCH * 20) == 0:
            print(f"  {split}: {seen} embedded", flush=True)

    embeddings = np.concatenate(vectors)
    group_array = np.concatenate(groups)
    is_hotdog = np.isin(group_array, [HOTDOG] if SCREEN_LABEL != "hotdog" else [HOTDOG, SCREENED])

    return {
        "embeddings": embeddings,
        "labels": is_hotdog.astype(np.float32),
        "food101Label": np.concatenate(labels),
        "group": group_array,
    }


def cache_path(split: str) -> "object":
    return CACHE_DIR / f"embeddings-{split}.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()

    ensure_dirs()
    for split in args.splits:
        easy = EASY_NEGATIVE_TAKE if split == "train" else EASY_NEGATIVE_TAKE // 4
        nonfood = NONFOOD_TAKE if split == "train" else NONFOOD_TAKE // 4
        data = embed_split(split, easy, nonfood)
        path = cache_path(split)
        np.savez_compressed(path, **data)
        positives = int(data["labels"].sum())
        total = len(data["labels"])
        print(
            f"{split}: {total} vectors of dim {data['embeddings'].shape[1]} "
            f"({positives} hotdog, {total - positives} not) -> {path}"
        )
        assert data["embeddings"].shape[1] == EMBEDDING_DIM


if __name__ == "__main__":
    main()
