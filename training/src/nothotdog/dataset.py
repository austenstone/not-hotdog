"""Turn Food-101 + Imagenette into a curated binary hotdog / not-hotdog dataset.

Accuracy here comes from the negatives, not the positives. Food-101 gives exactly 1000 hotdog
images and there is nothing to be done about that, so the work is in choosing what the model is
forced to distinguish them from.
"""

from __future__ import annotations

import json

import tensorflow as tf
import tensorflow_datasets as tfds

from .config import (
    ARTIFACT_DIR,
    EASY_NEGATIVE_TAKE,
    HARD_NEGATIVES,
    IMAGE_SIZE,
    IMAGENETTE_CONFIG,
    NONFOOD_TAKE,
    POSITIVE_CLASS,
    SEED,
    ensure_dirs,
)

AUTOTUNE = tf.data.AUTOTUNE

HOTDOG = 0
HARD_NEGATIVE = 1
EASY_NEGATIVE = 2
NONFOOD = 3

GROUP_NAMES = {
    HOTDOG: "hotdog",
    HARD_NEGATIVE: "hard_negative",
    EASY_NEGATIVE: "easy_negative",
    NONFOOD: "nonfood",
}


def food101_labels() -> list[str]:
    return tfds.builder("food101").info.features["label"].names


def _resize(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.shape(image)
    side = tf.minimum(shape[0], shape[1])
    image = tf.image.resize_with_crop_or_pad(image, side, side)
    return tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])


def _label_sets() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    names = food101_labels()
    index = {name: i for i, name in enumerate(names)}
    positive = tf.constant([index[POSITIVE_CLASS]], tf.int64)
    hard = tf.constant(sorted(index[n] for n in HARD_NEGATIVES), tf.int64)
    easy = tf.constant(
        sorted(i for name, i in index.items() if name != POSITIVE_CLASS and name not in HARD_NEGATIVES),
        tf.int64,
    )
    return positive, hard, easy


def _in_set(label: tf.Tensor, allowed: tf.Tensor) -> tf.Tensor:
    return tf.reduce_any(tf.equal(tf.cast(label, tf.int64), allowed))


def _food_subset(split: str, allowed: tf.Tensor, group: int) -> tf.data.Dataset:
    ds = tfds.load("food101", split=split, shuffle_files=True)
    ds = ds.filter(lambda x: _in_set(x["label"], allowed))
    return ds.map(
        lambda x: (_resize(x["image"]), tf.cast(x["label"], tf.int32), tf.constant(group, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )


def _nonfood(split: str, take: int) -> tf.data.Dataset:
    ds = tfds.load(IMAGENETTE_CONFIG, split=split, shuffle_files=True)
    ds = ds.map(
        lambda x: (_resize(x["image"]), tf.constant(-1, tf.int32), tf.constant(NONFOOD, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    return ds.take(take) if take > 0 else ds


def group_datasets(split: str, *, easy_take: int, nonfood_take: int) -> dict[int, tf.data.Dataset]:
    """The four curated groups, kept separate so each can be expanded differently.

    Yields (image[0,1], food101_label, group) per element.
    """
    positive, hard, easy = _label_sets()

    easy_negatives = _food_subset(split, easy, EASY_NEGATIVE)
    if easy_take > 0:
        easy_negatives = easy_negatives.shuffle(8192, seed=SEED).take(easy_take)

    nonfood_split = "train" if split.startswith("train") else "validation"

    return {
        HOTDOG: _food_subset(split, positive, HOTDOG),
        HARD_NEGATIVE: _food_subset(split, hard, HARD_NEGATIVE),
        EASY_NEGATIVE: easy_negatives,
        NONFOOD: _nonfood(nonfood_split, nonfood_take),
    }


def build_split(split: str, *, easy_take: int, nonfood_take: int) -> tf.data.Dataset:
    groups = group_datasets(split, easy_take=easy_take, nonfood_take=nonfood_take)
    combined = groups[HOTDOG]
    for key in (HARD_NEGATIVE, EASY_NEGATIVE, NONFOOD):
        combined = combined.concatenate(groups[key])
    return combined


def train_split() -> tf.data.Dataset:
    return build_split("train", easy_take=EASY_NEGATIVE_TAKE, nonfood_take=NONFOOD_TAKE)


def validation_split() -> tf.data.Dataset:
    return build_split("validation", easy_take=EASY_NEGATIVE_TAKE // 4, nonfood_take=NONFOOD_TAKE // 4)


def download() -> None:
    tfds.builder("food101").download_and_prepare()
    tfds.builder(IMAGENETTE_CONFIG).download_and_prepare()


def main() -> None:
    ensure_dirs()
    download()

    names = food101_labels()
    counts: dict[str, int] = {}
    for split_name, ds in (("train", train_split()), ("validation", validation_split())):
        per_group = [0, 0, 0, 0]
        for _, _, group in ds.map(lambda i, l, g: (0, 0, g)).as_numpy_iterator():
            per_group[int(group)] += 1
        for group, count in enumerate(per_group):
            counts[f"{split_name}/{GROUP_NAMES[group]}"] = count

    manifest = {
        "positiveClass": POSITIVE_CLASS,
        "hardNegatives": HARD_NEGATIVES,
        "easyNegativeClasses": [
            n for n in names if n != POSITIVE_CLASS and n not in HARD_NEGATIVES
        ],
        "nonfoodSource": IMAGENETTE_CONFIG,
        "counts": counts,
    }
    path = ARTIFACT_DIR / "dataset-manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
