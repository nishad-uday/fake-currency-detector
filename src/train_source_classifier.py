
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "authenticity_ready"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

OUTPUT_MODEL = MODEL_DIR / "currency_source_classifier.pth"
OUTPUT_REPORT = REPORT_DIR / "source_classifier_report.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
SEED = 42

SOURCE_FOLDERS = [
    "user_photos_unverified",
    "repository_fake_unverified",
    "repository_reference_unverified",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def count_images(folder):
    if not folder.is_dir():
        return 0
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

def main():
    print("=" * 55)
    print("EXPERIMENTAL SOURCE CLASSIFIER")
    print("This is NOT a counterfeit detector.")
    print("=" * 55)
    print("Device:", DEVICE)

    for name in SOURCE_FOLDERS:
        folder = DATA_ROOT / name
        count = count_images(folder)
        print(f"{name}: {count} images")
        if count < 2:
            raise SystemExit(f"Not enough images in {folder}")

    # Random split is only exploratory. Similar images from the
    # same physical note/source may leak across train and validation.
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

  
    from torch.utils.data import Dataset
    from PIL import Image

    class SourceDataset(Dataset):
        def __init__(self, root, class_names, transform):
            self.transform = transform
            self.classes = class_names
            self.class_to_idx = {
                name: i for i, name in enumerate(class_names)
            }
            self.samples = []

            extensions = {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp"
            }

            for class_name in class_names:
                folder = root / class_name

                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix.lower() in extensions:
                        self.samples.append(
                            (path, self.class_to_idx[class_name])
                        )

            self.targets = [label for _, label in self.samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, label = self.samples[index]
            with Image.open(path) as image:
                image = image.convert("RGB")
            return self.transform(image), label

    dataset = SourceDataset(
        DATA_ROOT, SOURCE_FOLDERS, transform
    )


    # Keep only the three source folders, excluding train_data,
    # validation_data and any other directories.
    allowed_indices = [
        i for i, (_, label) in enumerate(dataset.samples)
        if dataset.classes[label] in SOURCE_FOLDERS
    ]
    dataset.samples = [dataset.samples[i] for i in allowed_indices]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]

    if not dataset.samples:
        raise SystemExit("No source images found.")

    generator = torch.Generator().manual_seed(SEED)
    val_size = max(3, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    if train_size < 3:
        raise SystemExit("Not enough images for a train/validation split.")

    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as error:
        raise SystemExit(
            "Could not load pretrained ResNet18 weights. "
            "Internet may be needed for the initial download.\n"
            f"Details: {error}"
        )

    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                predictions = model(images).argmax(1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / max(total, 1):.4f} | "
            f"Train accuracy: {100 * correct / max(total, 1):.2f}% | "
            f"Validation accuracy: {100 * val_correct / max(val_total, 1):.2f}%"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "image_size": IMAGE_SIZE,
        "purpose": "Source classification only; not authenticity detection.",
    }, OUTPUT_MODEL)

    report = {
        "purpose": "Source classification only, not fake currency detection",
        "classes": dataset.classes,
        "images": len(dataset),
        "train_images": train_size,
        "validation_images": val_size,
        "last_validation_accuracy_percent": round(
            100 * val_correct / max(val_total, 1), 2
        ),
        "limitations": [
            "Source folders have unverified authenticity labels.",
            "Random splitting may leak related images.",
            "Validation accuracy does not measure counterfeit detection.",
        ],
    }

    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nSaved model:", OUTPUT_MODEL)
    print("Saved report:", OUTPUT_REPORT)
    print("Reminder: this model does NOT predict genuine or fake.")

if __name__ == "__main__":
    main()
