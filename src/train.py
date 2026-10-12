
from pathlib import Path
import json
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed_mendeley"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "currency_resnet18.pth"
BACKUP_PATH = MODEL_DIR / "currency_resnet18_before_aug.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main():
    set_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = DATA_DIR / "train"

    # Your existing dataset split uses a validation folder.
    # Support either "val" or "valid".
    if (DATA_DIR / "val").exists():
        val_dir = DATA_DIR / "val"
    elif (DATA_DIR / "valid").exists():
        val_dir = DATA_DIR / "valid"
    else:
        raise FileNotFoundError(
            f"Validation folder not found under: {DATA_DIR}\n"
            "Expected a folder named 'val' or 'valid'."
        )

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}"
        )

    # --------------------------------------
    # BACKUP EXISTING TRAINED MODEL
    # --------------------------------------

    if MODEL_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Previous model backed up to:\n{BACKUP_PATH}")

    # --------------------------------------
    # IMAGE TRANSFORMS
    # --------------------------------------

    # Mild augmentation: preserve important note details.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomAffine(
            degrees=8,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
            fill=0
        ),

        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0.08,
            hue=0.01
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ])

    # Validation images must remain unaugmented.
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        ),
    ])

    train_data = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform
    )

    val_data = datasets.ImageFolder(
        root=str(val_dir),
        transform=val_transform
    )

    if train_data.classes != val_data.classes:
        raise ValueError(
            "Training and validation classes do not match.\n"
            f"Train: {train_data.classes}\n"
            f"Validation: {val_data.classes}"
        )

    if len(train_data) == 0 or len(val_data) == 0:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    print("=" * 55)
    print("INDIAN CURRENCY DENOMINATION TRAINING")
    print("=" * 55)
    print("Classes:", train_data.classes)
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Augmentation: mild rotation, translation, color")
    print("Test set will not be used during training.")
    print("=" * 55)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------
    # MODEL
    # --------------------------------------

    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(train_data.classes)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    best_val_acc = -1.0
    history = []

    # --------------------------------------
    # TRAINING LOOP
    # --------------------------------------

    for epoch in range(EPOCHS):
        start_time = time.time()

        # -------- TRAIN --------
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # -------- VALIDATION --------
        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_sum += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": current_lr,
            "time_seconds": round(elapsed, 2)
        }

        history.append(record)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.2%} | "
            f"LR: {current_lr:.6g} | "
            f"Time: {elapsed:.1f}s"
        )

        # ----------------------------------
        # SAVE BEST VALIDATION MODEL
        # ----------------------------------

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes,
                "image_size": IMAGE_SIZE,
                "architecture": "resnet18",
                "best_val_accuracy": best_val_acc,
                "epoch": epoch + 1
            }

            torch.save(checkpoint, MODEL_PATH)

            with open(
                MODEL_DIR / "classes.json",
                "w",
                encoding="utf-8"
            ) as file:
                json.dump(
                    train_data.classes,
                    file,
                    indent=2
                )

            print(
                f"Best model saved at epoch {epoch + 1}."
            )

        # Save history after every epoch.
        with open(
            MODEL_DIR / "training_history.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(history, file, indent=2)

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved at: {MODEL_PATH}")
    print(
        f"Training history: "
        f"{MODEL_DIR / 'training_history.json'}"
    )

    if BACKUP_PATH.exists():
        print(f"Previous model backup: {BACKUP_PATH}")

    print("\nNext: evaluate the saved model on the test set.")
    print("Training accuracy is not the same as test accuracy.")


if __name__ == "__main__":
    main()
