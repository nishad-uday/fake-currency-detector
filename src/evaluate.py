
from pathlib import Path
import csv
import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT / "dataset" / "processed_mendeley" / "test"
MODEL_PATH = ROOT / "models" / "currency_resnet18.pth"
OUTPUT_DIR = ROOT / "models" / "evaluation"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def normalize_label(value):
    """Normalize denomination labels for folder/checkpoint comparison."""
    text = str(value).strip()
    text = text.replace("₹", "").replace("Rs.", "").replace("Rs", "")
    text = text.replace("rupees", "").replace("Rupees", "").strip()

    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_trained_model():
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset folder not found:\n{TEST_DIR}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    print("Loading trained model...")
    print(f"Device: {DEVICE}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if "classes" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'classes'. "
            "Check the checkpoint format used by train.py."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'. "
            "Check the checkpoint format used by train.py."
        )

    class_names = [str(name) for name in checkpoint["classes"]]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def main():
    try:
        model, class_names = load_trained_model()
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    dataset = ImageFolder(
        root=str(TEST_DIR),
        transform=TRANSFORM
    )

    if len(dataset) == 0:
        print("ERROR: No test images found.")
        sys.exit(1)

    # Match test folder classes to the trained model's class order.
    folder_classes = dataset.classes

    model_label_map = {
        normalize_label(name): index
        for index, name in enumerate(class_names)
    }

    folder_label_map = {
        normalize_label(name): index
        for index, name in enumerate(folder_classes)
    }

    if set(model_label_map) != set(folder_label_map):
        print("ERROR: Test folder classes do not match model classes.")
        print("Model classes:", class_names)
        print("Test classes:", folder_classes)
        sys.exit(1)

    # Convert ImageFolder targets into the model's class order.
    model_targets = [
        model_label_map[normalize_label(folder_classes[target])]
        for _, target in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    true_labels = []
    predicted_labels = []
    predicted_scores = []
    image_paths = []

    print(f"\nTest images: {len(dataset)}")
    print(f"Classes: {class_names}")
    print("\nEvaluating...")

    sample_offset = 0

    with torch.no_grad():
        for images, _folder_targets in loader:
            images = images.to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            scores, predictions = probabilities.max(dim=1)

            batch_size = images.size(0)

            true_labels.extend(
                model_targets[sample_offset:sample_offset + batch_size]
            )
            predicted_labels.extend(predictions.cpu().tolist())
            predicted_scores.extend(scores.cpu().tolist())

            image_paths.extend([
                str(dataset.samples[i][0])
                for i in range(
                    sample_offset,
                    sample_offset + batch_size
                )
            ])

            sample_offset += batch_size

    accuracy = accuracy_score(true_labels, predicted_labels)

    # Human-readable classification report.
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    # Save text report.
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("INDIAN CURRENCY DENOMINATION MODEL EVALUATION\n")
        file.write("=" * 55 + "\n")
        file.write(f"Test images: {len(dataset)}\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        file.write(report_text)

    # Save metrics CSV.
    metrics_path = OUTPUT_DIR / "metrics.csv"

    with open(
        metrics_path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "denomination",
            "precision",
            "recall",
            "f1_score",
            "support"
        ])

        for name in class_names:
            row = report[name]
            writer.writerow([
                name,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                int(row["support"])
            ])

        writer.writerow([])
        writer.writerow(["overall_accuracy", f"{accuracy:.4f}"])

    # Confusion matrix.
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names)))
    )

    figure, axis = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8))
    )

    image = axis.imshow(cm, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=[f"₹{name}" for name in class_names],
        yticklabels=[f"₹{name}" for name in class_names],
        xlabel="Predicted denomination",
        ylabel="Actual denomination",
        title="Confusion Matrix - Test Dataset"
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor"
    )

    threshold = cm.max() / 2 if cm.size and cm.max() else 0

    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            axis.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                color="white" if cm[row, column] > threshold else "black"
            )

    figure.tight_layout()

    matrix_path = OUTPUT_DIR / "confusion_matrix.png"
    figure.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    # Save wrong predictions.
    wrong_path = OUTPUT_DIR / "wrong_predictions.csv"

    with open(
        wrong_path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path",
            "actual",
            "predicted",
            "model_score_percent"
        ])

        for path, actual, predicted, score in zip(
            image_paths,
            true_labels,
            predicted_labels,
            predicted_scores
        ):
            if actual != predicted:
                writer.writerow([
                    path,
                    class_names[actual],
                    class_names[predicted],
                    round(score * 100, 2)
                ])

    # Print summary.
    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total test images : {len(dataset)}")
    print(f"Correct           : {sum(a == b for a, b in zip(true_labels, predicted_labels))}")
    print(f"Incorrect         : {sum(a != b for a, b in zip(true_labels, predicted_labels))}")
    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print("\nClassification report:\n")
    print(report_text)

    print("\nFiles saved:")
    print(f"Report          : {report_path}")
    print(f"Metrics CSV     : {metrics_path}")
    print(f"Confusion matrix: {matrix_path}")
    print(f"Wrong predictions: {wrong_path}")

    print("\nNote: These metrics evaluate denomination prediction only.")
    print("They do not measure genuine/counterfeit detection.")


if __name__ == "__main__":
    main()
