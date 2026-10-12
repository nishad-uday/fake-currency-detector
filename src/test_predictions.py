
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parents[1]

TEST_ROOT = ROOT / "dataset" / "processed_mendeley" / "test"
MODEL_PATH = ROOT / "models" / "currency_resnet18.pth"

CLASSES = ["10", "20", "50", "100", "200", "500", "2000"]
IMAGES_PER_CLASS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model_classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(model_classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, model_classes


def get_test_images():
    if not TEST_ROOT.is_dir():
        raise FileNotFoundError(
            f"Test dataset folder not found: {TEST_ROOT}"
        )

    selected_images = []

    for denomination in CLASSES:
        folder = TEST_ROOT / denomination

        if not folder.is_dir():
            print(f"Warning: Missing folder for ₹{denomination}")
            continue

        files = sorted(
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not files:
            print(f"Warning: No images found for ₹{denomination}")
            continue

        selected = files[:IMAGES_PER_CLASS]

        for path in selected:
            selected_images.append((path, denomination))

        print(
            f"₹{denomination}: "
            f"{len(selected)}/{IMAGES_PER_CLASS} images selected"
        )

    return selected_images


def main():
    print("Loading trained model...")
    print(f"Device: {DEVICE}")

    model, model_classes = load_model()
    images = get_test_images()

    if not images:
        print("No test images available.")
        return

    correct = 0
    total = 0
    class_results = {
        denomination: {"correct": 0, "total": 0}
        for denomination in CLASSES
    }

    print("\n" + "=" * 75)
    print("DENOMINATION PREDICTIONS")
    print("=" * 75)

    for image_path, actual in images:
        try:
            with Image.open(image_path) as image:
                tensor = TRANSFORM(
                    image.convert("RGB")
                ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(tensor)
                scores = torch.softmax(outputs, dim=1)[0]

            top_values, top_indices = torch.topk(
                scores,
                k=min(3, len(model_classes))
            )

            predicted = model_classes[
                top_indices[0].item()
            ]

            is_correct = predicted == actual

            total += 1
            class_results[actual]["total"] += 1

            if is_correct:
                correct += 1
                class_results[actual]["correct"] += 1

            status = "CORRECT" if is_correct else "WRONG"

            print(f"\nImage: {image_path.name}")
            print(f"Actual: ₹{actual}")
            print(f"Predicted: ₹{predicted} [{status}]")
            print(f"Top score: {top_values[0].item():.2%}")

            print("Top 3:")
            for value, index in zip(top_values, top_indices):
                denomination = model_classes[index.item()]
                print(f"  ₹{denomination}: {value.item():.2%}")

        except Exception as error:
            print(f"\nCould not process {image_path.name}: {error}")

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    if total:
        print(f"Correct predictions: {correct}/{total}")
        print(f"Accuracy on selected images: {correct / total:.2%}")

    print("\nPer-denomination results:")
    for denomination in CLASSES:
        result = class_results[denomination]
        if result["total"]:
            accuracy = result["correct"] / result["total"]
            print(
                f"₹{denomination}: "
                f"{result['correct']}/{result['total']} "
                f"({accuracy:.2%})"
            )

    print("\nNote:")
    print("- This is a small diagnostic sample, not final evaluation.")
    print("- Scores are not guaranteed probabilities.")
    print("- This model predicts denomination, not genuine/fake status.")


if __name__ == "__main__":
    main()
