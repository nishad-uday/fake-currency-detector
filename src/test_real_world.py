
from pathlib import Path
import csv
import re

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "dataset" / "real_world_test"
MODEL_PATH = ROOT / "models" / "currency_resnet18.pth"

OUTPUT_DIR = ROOT / "models" / "evaluation"
REPORT_PATH = OUTPUT_DIR / "real_world_predictions.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# READ LABELS FROM FILENAMES
# ==========================================

def actual_label_from_filename(filename):
    """
    Supported examples:
    10_front_01.png
    20_back_01.png
    50_front_01.jpeg
    100_back_01.jpeg
    200_front_01.png
    500_back_01.jpeg
    2000_front_01.png
    """

    stem = Path(filename).stem

    match = re.fullmatch(
        r"(10|20|50|100|200|500|2000)_(front|back)_\d+",
        stem,
        flags=re.IGNORECASE
    )

    if not match:
        return None, None

    denomination = match.group(1)
    side = match.group(2).lower()

    return denomination, side


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    classes = [str(item) for item in checkpoint["classes"]]

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model, classes


# ==========================================
# PREDICT ONE IMAGE
# ==========================================

def predict_image(model, classes, image_path):
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0]

    top_count = min(3, len(classes))

    top_scores, top_indices = torch.topk(
        scores,
        k=top_count
    )

    predicted_index = int(top_indices[0].item())
    predicted = classes[predicted_index]

    top_predictions = []

    for score, index in zip(
        top_scores.cpu().tolist(),
        top_indices.cpu().tolist()
    ):
        top_predictions.append(
            f"₹{classes[index]}: {score * 100:.2f}%"
        )

    top_score = float(top_scores[0].item()) * 100

    return predicted, top_score, " | ".join(top_predictions)


# ==========================================
# MAIN TESTING
# ==========================================

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"Test image folder not found:\n{IMAGE_DIR}"
        )

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )

    if not image_paths:
        print("No supported images found in the test folder.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    model, classes = load_model()

    print("Device:", DEVICE)
    print("Model classes:", classes)
    print("Images found:", len(image_paths))
    print("\nRunning online/real-world image testing...\n")

    results = []
    skipped = []

    for image_path in image_paths:
        actual, side = actual_label_from_filename(
            image_path.name
        )

        if actual is None:
            print(
                f"Skipping unrecognized filename: "
                f"{image_path.name}"
            )
            skipped.append(image_path.name)
            continue

        # Do not count labels the model was not trained to predict.
        if actual not in classes:
            print(
                f"Skipping unsupported model class: "
                f"{image_path.name} (₹{actual})"
            )
            skipped.append(image_path.name)
            continue

        try:
            predicted, score, top3 = predict_image(
                model,
                classes,
                image_path
            )

            correct = predicted == actual

            result = {
                "filename": image_path.name,
                "side": side,
                "actual": actual,
                "predicted": predicted,
                "correct": correct,
                "model_score_percent": round(score, 2),
                "top_3_predictions": top3
            }

            results.append(result)

            status = "CORRECT" if correct else "WRONG"

            print(
                f"{image_path.name} | "
                f"Actual: ₹{actual} | "
                f"Predicted: ₹{predicted} | "
                f"Score: {score:.2f}% | {status}"
            )

        except Exception as error:
            print(
                f"Could not process {image_path.name}: {error}"
            )
            skipped.append(image_path.name)

    # Save the report, including a header if no images were valid.
    fields = [
        "filename",
        "side",
        "actual",
        "predicted",
        "correct",
        "model_score_percent",
        "top_3_predictions"
    ]

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 58)
    print("ONLINE / REAL-WORLD TEST SUMMARY")
    print("=" * 58)

    if not results:
        print("No images were evaluated.")
        print("Check filenames and model class labels.")
    else:
        for side in ["front", "back"]:
            side_results = [
                item for item in results
                if item["side"] == side
            ]

            if side_results:
                correct = sum(
                    item["correct"] for item in side_results
                )
                total = len(side_results)

                print(
                    f"{side.title():<6}: "
                    f"{correct}/{total} correct "
                    f"({correct / total * 100:.2f}%)"
                )

        correct_total = sum(
            item["correct"] for item in results
        )
        total = len(results)

        print(
            f"\nOverall: {correct_total}/{total} correct "
            f"({correct_total / total * 100:.2f}%)"
        )

    print("Images evaluated:", len(results))
    print("Images skipped:", len(skipped))
    print(f"\nCSV report saved to:\n{REPORT_PATH}")

    print("\nIMPORTANT:")
    print("- This evaluates denomination classification only.")
    print("- It does not verify genuine or counterfeit notes.")
    print("- Online images and camera photos should be reported separately.")
    print("- Scores are model outputs, not guaranteed probabilities.")


if __name__ == "__main__":
    main()
