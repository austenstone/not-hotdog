"""Augmentation, including synthesized photo-of-a-screen effects.

Food-101 is professionally shot, well-lit food. Phone photos are none of those things, so this
module is the bridge between the two and is the highest-leverage thing to tune if real-world
accuracy disappoints.
"""

from __future__ import annotations

import math

import tensorflow as tf

from .config import IMAGE_SIZE


def _rotate(image: tf.Tensor, radians: tf.Tensor) -> tf.Tensor:
    cos, sin = tf.cos(radians), tf.sin(radians)
    size = tf.cast(tf.shape(image)[0], tf.float32)
    center = (size - 1.0) / 2.0
    offset_x = center - (cos * center - sin * center)
    offset_y = center - (sin * center + cos * center)
    transform = tf.stack([cos, -sin, offset_x, sin, cos, offset_y, 0.0, 0.0])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis],
        transforms=transform[tf.newaxis],
        output_shape=tf.shape(image)[:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]


def _channel_shift(image: tf.Tensor, intensity: float, seed: tf.Tensor) -> tf.Tensor:
    shift = tf.random.stateless_uniform([3], seed=seed, minval=-intensity, maxval=intensity)
    return image + shift


def augment(image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Geometric and photometric jitter. Expects float32 in [0, 1], returns the same."""
    seeds = tf.random.stateless_uniform([6, 2], seed=seed, maxval=2**30, dtype=tf.int32)

    image = tf.image.stateless_random_flip_left_right(image, seeds[0])

    angle = tf.random.stateless_uniform([], seed=seeds[1], minval=-0.35, maxval=0.35)
    image = _rotate(image, angle)

    scale = tf.random.stateless_uniform([], seed=seeds[2], minval=0.75, maxval=1.0)
    crop = tf.cast(tf.round(scale * IMAGE_SIZE), tf.int32)
    image = tf.image.stateless_random_crop(image, [crop, crop, 3], seeds[3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    image = tf.image.stateless_random_brightness(image, 0.25, seeds[4])
    image = tf.image.stateless_random_contrast(image, 0.7, 1.4, seeds[5])
    image = _channel_shift(image, 0.12, seeds[4])

    return tf.clip_by_value(image, 0.0, 1.0)


def screen_effect(image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """Simulate photographing a hotdog off an LCD panel.

    Users tried to cheat the original app this way. Rather than scrape photos of monitors we
    synthesize the artefacts it produces: pixel-grid moire, scanlines, glare and a dark bezel.
    """
    seeds = tf.random.stateless_uniform([5, 2], seed=seed, maxval=2**30, dtype=tf.int32)

    coords = tf.cast(tf.range(IMAGE_SIZE), tf.float32)
    xs = coords[tf.newaxis, :, tf.newaxis]
    ys = coords[:, tf.newaxis, tf.newaxis]

    freq = tf.random.stateless_uniform([], seed=seeds[0], minval=0.55, maxval=1.5)
    angle = tf.random.stateless_uniform([], seed=seeds[1], minval=0.0, maxval=math.pi)
    projected = xs * tf.cos(angle) + ys * tf.sin(angle)
    moire = tf.sin(projected * freq) * tf.sin(xs * freq * 0.97) * 0.09
    image = image + moire

    scanlines = tf.sin(ys * math.pi) * 0.05
    image = image - scanlines

    gx = tf.random.stateless_uniform([], seed=seeds[2], minval=-1.0, maxval=1.0)
    gy = tf.random.stateless_uniform([], seed=seeds[3], minval=-1.0, maxval=1.0)
    norm = tf.cast(IMAGE_SIZE, tf.float32)
    glare = tf.clip_by_value((xs / norm) * gx + (ys / norm) * gy, 0.0, 1.0)
    strength = tf.random.stateless_uniform([], seed=seeds[4], minval=0.1, maxval=0.35)
    image = image * (1.0 - strength * 0.5) + glare * strength

    bezel = tf.random.stateless_uniform([], seed=seeds[0], minval=0.0, maxval=0.12)
    inset = tf.cast(tf.round(bezel * IMAGE_SIZE), tf.int32)
    inner = IMAGE_SIZE - 2 * inset
    image = tf.image.resize(image, [inner, inner])
    image = tf.image.pad_to_bounding_box(image, inset, inset, IMAGE_SIZE, IMAGE_SIZE)

    return tf.clip_by_value(image, 0.0, 1.0)


def to_model_input(image: tf.Tensor) -> tf.Tensor:
    """[0, 1] float -> MobileNetV2's expected [-1, 1]."""
    return image * 2.0 - 1.0
