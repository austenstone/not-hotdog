"""Shared configuration for the Not Hotdog training pipeline."""

from __future__ import annotations

import os
from pathlib import Path

IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
EMBEDDING_DIM = 1280

POSITIVE_CLASS = "hot_dog"
CLASS_ORDER = ["not_hotdog", "hotdog"]

# Food-101 classes chosen because they are what a naive classifier actually confuses with a
# hotdog: split buns, cylindrical wraps, and red-brown meat. lobster_roll_sandwich and
# breakfast_burrito are the two worst offenders.
HARD_NEGATIVES = [
    "lobster_roll_sandwich",
    "pulled_pork_sandwich",
    "breakfast_burrito",
    "hamburger",
    "club_sandwich",
    "grilled_cheese_sandwich",
    "croque_madame",
    "garlic_bread",
    "bruschetta",
    "spring_rolls",
    "tacos",
    "sushi",
    "gyoza",
    "samosa",
    "baby_back_ribs",
    "chicken_wings",
    "pork_chop",
    "french_fries",
    "poutine",
    "onion_rings",
    "carrot_cake",
]

# Easy negatives are subsampled while hard negatives are kept whole. That asymmetry *is* the
# oversampling: hard negatives end up far over-represented relative to their natural frequency,
# without duplicating a single image.
EASY_NEGATIVE_TAKE = 20_000
NONFOOD_TAKE = 8_000

# Food-101 is exactly balanced: every one of the 101 classes has 750 train and 250 validation
# images. Group sizes are therefore known statically, with no pass over the data required.
PER_CLASS_TRAIN = 750
PER_CLASS_VALIDATION = 250

# The original app trained on roughly 3k hotdogs. 750 Food-101 hotdogs x 4 deterministic
# augmented variants lands in the same place.
POSITIVE_VARIANTS = 4
SCREEN_VARIANTS = 2

# "hotdog" keeps a hotdog-through-a-screen classified as a hotdog (robustness).
# "not_hotdog" reproduces the show's anti-cheat spirit.
SCREEN_LABEL = os.environ.get("SCREEN_LABEL", "hotdog")

IMAGENETTE_CONFIG = "imagenette/320px-v2"

TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent
CACHE_DIR = Path(os.environ.get("NOTHOTDOG_CACHE", TRAINING_ROOT / "cache"))
ARTIFACT_DIR = Path(os.environ.get("NOTHOTDOG_ARTIFACTS", TRAINING_ROOT / "artifacts"))
MODELS_DIR = REPO_ROOT / "models"

TARGET_PRECISION = 0.97
SEED = 1337


def ensure_dirs() -> None:
    for d in (CACHE_DIR, ARTIFACT_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
