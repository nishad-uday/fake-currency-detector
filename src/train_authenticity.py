
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "authenticity_ready"

TRAIN_DIR = DATA_ROOT / "train_data"
VAL_DIR = DATA_ROOT / "validation_data"

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = MODEL_DIR / "currency_authenticity_resnet18.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5
IMAGE_SIZE = 224

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def count_images(folder):
    if not folder.exists():
        return 0

    return sum(
        1 for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_folders():
    expected = ["genuine", "fake"]

    for folder in [TRAIN_DIR, VAL_DIR]:
        if not folder.is_dir():
            print(f"ERROR: Folder missing: {folder}")
            return False

        for class_name in expected:
            class_folder = folder / class_name

            if not class_folder.is_dir():
                print(f"ERROR: Folder missing: {class_folder}")
                return False

            count = count_images(class_folder)
            print(f"{folder.name}/{class_name}: {count} images")

            if count == 0:
                print(
                    "\nSTOP: Genuine/fake folders need correctly labelled "
                    "images before training."
                )
                return False

    return True


def main():
    print("=" * 55)
    print("AUTHENTICITY MODEL TRAINING")
    print("=" * 55)
    print("Device:", DEVICE)
    print("Existing denomination model will not be changed.")
    print()

    if not validate_folders():
        print("\nTraining not started. No model was created.")
        print(
            "Only use images whose genuine/fake labels have been "
            "independently verified."
        )
        return

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_transform
    )

    print("\nTraining classes:", train_dataset.classes)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    if train_dataset.classes != ["fake", "genuine"]:
        print("ERROR: Expected class folders named fake and genuine.")
        return

    if val_dataset.classes != train_dataset.classes:
        print("ERROR: Training and validation classes do not match.")
        return

    # Prevent meaningless training with a single example per class.
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    if min(train_counts.values(), default=0) < 2:
        print("ERROR: Need at least 2 images in each training class.")
        return

    if min(val_counts.values(), default=0) < 1:
        print("ERROR: Need validation images in both classes.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    try:
        model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )
    except Exception as error:
        print("\nCould not load pretrained weights:", error)
        print(
            "Training cancelled. Internet may be needed to download "
            "ResNet18 pretrained weights."
        )
        return

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    best_accuracy = -1.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
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

            running_loss += loss.item() * images.size(0)
            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        train_loss = running_loss / max(train_total, 1)
        train_accuracy = 100 * train_correct / max(train_total, 1)
        val_accuracy = 100 * val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2f}% | "
            f"Validation accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "validation_accuracy": val_accuracy,
                "warning": (
                    "Use only if labels are verified. Validation accuracy "
                    "does not prove real-world counterfeit detection."
                )
            }, OUTPUT_PATH)

    print("\nTraining finished.")
    print("Model saved to:", OUTPUT_PATH)
    print("Best validation accuracy:", round(best_accuracy, 2), "%")
    print(
        "Important: Accuracy is meaningful only if labels are correct "
        "and validation images are independent."
    )


if __name__ == "__main__":
    main()
