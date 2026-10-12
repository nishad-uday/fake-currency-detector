from pathlib import Path
import torch


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Dataset paths
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATASET_DIR = DATASET_DIR / "raw"
PROCESSED_DATASET_DIR = DATASET_DIR / "processed"


# Model directory
MODEL_DIR = PROJECT_ROOT / "models"


# Automatically select GPU or CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Image settings
IMAGE_SIZE = 224


# Training settings
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42


if __name__ == "__main__":

    print("Project Root:", PROJECT_ROOT)
    print("Dataset:", DATASET_DIR)
    print("Raw Dataset:", RAW_DATASET_DIR)
    print("Processed Dataset:", PROCESSED_DATASET_DIR)
    print("Model Directory:", MODEL_DIR)
    print("Device:", DEVICE)