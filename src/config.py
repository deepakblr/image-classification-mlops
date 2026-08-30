"""Project-wide configuration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Microsoft Cats & Dogs (same images as the Kaggle Cats vs Dogs challenge)
DATASET_URL = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
DATASET_ZIP_NAME = "kagglecatsanddogs_5340.zip"

IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ("cat", "dog")
CLASS_TO_IDX = {"cat": 0, "dog": 1}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

CHAMPION_MODEL_PATH = MODELS_DIR / "champion_cnn.pt"
SPLITS_PATH = PROCESSED_DIR / "splits.json"

# Training defaults
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
SMOKE_EPOCHS = 1
SMOKE_MAX_PER_CLASS = 32
SMOKE_BATCH_SIZE = 8

EXPERIMENT_NAME = "cats-vs-dogs-classification"
REGISTERED_MODEL_NAME = "cats-dogs-cnn"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
