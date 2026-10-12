
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "authenticity_ready"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "currency_authenticity_experimental.pth"
REPORT_PATH = REPORT_DIR / "authenticity_experimental_report.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 4
SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# These are repository claims, NOT independently verified labels.
SOURCES = [
    ("repository_fake_unverified", 0),
    ("repository_reference_unverified", 1),
]
CLASS_NAMES = ["repository_claimed_fake", "repository_reference_unverified"]

class NoteDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        return self.transform(image), label

def main():
    samples = []

    for folder_name, label in SOURCES:
        folder = DATA / folder_name
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        print(f"{folder_name}: {len(files)} images")
        samples.extend((p, label) for p in files)

    counts = [sum(label == i for _, label in samples) for i in range(2)]
    if min(counts) < 5:
        raise SystemExit("Not enough images for an exploratory split.")

    indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=SEED,
        stratify=[label for _, label in samples],
    )

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_data = NoteDataset([samples[i] for i in train_idx], transform_train)
    val_data = NoteDataset([samples[i] for i in val_idx], transform_val)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Device:", DEVICE)
    print("Train images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Loading pretrained ResNet18...")

    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        raise SystemExit(
            f"Could not load pretrained weights: {exc}\n"
            "Check internet connection and run again."
        )

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / total:.4f} | "
            f"Train accuracy: {100 * correct / total:.2f}%"
        )

    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(1).cpu().tolist()
            actual.extend(labels.tolist())
            predicted.extend(predictions)

    val_accuracy = 100 * sum(
        a == p for a, p in zip(actual, predicted)
    ) / len(actual)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "classes": CLASS_NAMES,
        "image_size": 224,
        "purpose": "Experimental classification of unverified repository labels only",
    }, MODEL_PATH)

    report = {
        "purpose": "Experimental; not validated counterfeit detection",
        "classes": CLASS_NAMES,
        "images": len(samples),
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "validation_accuracy_percent": round(val_accuracy, 2),
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            actual, predicted,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "limitations": [
            "Source labels are unverified repository claims.",
            "Random split may contain related images in both sets.",
            "Validation accuracy does not measure real-world counterfeit detection.",
        ],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nValidation accuracy:", round(val_accuracy, 2), "%")
    print("Model saved:", MODEL_PATH)
    print("Report saved:", REPORT_PATH)
    print("This is NOT yet a reliable genuine/fake detector.")

if __name__ == "__main__":
    main()
