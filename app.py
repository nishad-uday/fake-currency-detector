
from pathlib import Path
import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageStat, ImageFilter, UnidentifiedImageError
from flask import Flask, render_template, request

# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "currency_resnet18.pth"
AUTHENTICITY_MODEL_PATH = (
    ROOT / "models" / "currency_authenticity_experimental.pth"
)
REFERENCE_ROOT = ROOT / "dataset" / "processed_mendeley"
UPLOAD_DIR = ROOT / "static" / "uploads"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 70.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# MODEL LOADER
# =========================================================

def load_resnet18_checkpoint(model_path, description):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if "classes" not in checkpoint:
        raise ValueError(f"Classes missing in {description} checkpoint.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Model weights missing in {description} checkpoint."
        )

    class_names = checkpoint["classes"]

    network = models.resnet18(weights=None)
    network.fc = nn.Linear(
        network.fc.in_features,
        len(class_names),
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.to(DEVICE)
    network.eval()

    return network, class_names


# =========================================================
# LOAD DENOMINATION MODEL
# =========================================================

model, classes = load_resnet18_checkpoint(
    MODEL_PATH,
    "Denomination model",
)

print("Denomination model loaded.")
print("Denomination classes:", classes)


# =========================================================
# LOAD EXPERIMENTAL AUTHENTICITY MODEL
# =========================================================

authenticity_model = None
authenticity_classes = None


def initialize_authenticity_model():
    global authenticity_model, authenticity_classes

    if not AUTHENTICITY_MODEL_PATH.is_file():
        print("Experimental authenticity model not found.")
        return

    try:
        authenticity_model, authenticity_classes = (
            load_resnet18_checkpoint(
                AUTHENTICITY_MODEL_PATH,
                "Experimental authenticity model",
            )
        )

        print("Experimental model loaded.")
        print("Experimental classes:", authenticity_classes)

    except (OSError, RuntimeError, ValueError, KeyError) as error:
        authenticity_model = None
        authenticity_classes = None
        print("Experimental model could not be loaded:", error)


# initialize_authenticity_model()

torch.set_num_threads(1)


# =========================================================
# REFERENCE-BASED VISUAL SIMILARITY
# This is NOT verified counterfeit detection.
# =========================================================

REFERENCE_BANK = {}
REFERENCE_THRESHOLDS = {}

feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
).to(DEVICE)
feature_extractor.eval()


def get_visual_embedding(image):
    """Extract normalized visual features from the denomination model."""
    tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = feature_extractor(tensor).flatten(1)
        features = F.normalize(features, p=2, dim=1)

    return features[0].cpu()


def list_image_files(folder):
    if not folder.is_dir():
        return []

    allowed = {".jpg", ".jpeg", ".png", ".webp"}

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )


def build_reference_bank():
    """
    Build a visual reference bank from train images.
    Validation similarities create provisional thresholds only.
    """

    global REFERENCE_BANK, REFERENCE_THRESHOLDS

    train_dir = REFERENCE_ROOT / "train"
    val_dir = REFERENCE_ROOT / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        print("Reference dataset unavailable; screening disabled.")
        return

    # Load training reference embeddings.
    for class_name in classes:
        key = str(class_name)
        class_dir = train_dir / key
        embeddings = []

        for path in list_image_files(class_dir)[:8]:
            try:
                with Image.open(path) as img:
                    embeddings.append(get_visual_embedding(img))
            except (OSError, ValueError, RuntimeError) as error:
                print(f"Skipping reference {path.name}: {error}")

        if embeddings:
            REFERENCE_BANK[key] = torch.stack(embeddings)

    # Calculate similarities for validation images.
    validation_scores = {
        str(class_name): [] for class_name in classes
    }

    for class_name in classes:
        key = str(class_name)
        references = REFERENCE_BANK.get(key)

        if references is None:
            continue

        for path in list_image_files(val_dir / key)[:5]:
            try:
                with Image.open(path) as img:
                    embedding = get_visual_embedding(img)

                similarity = float(
                    (references @ embedding).max().item()
                )
                validation_scores[key].append(similarity)

            except (OSError, ValueError, RuntimeError) as error:
                print(f"Skipping validation {path.name}: {error}")

    # Provisional threshold, not a counterfeit/genuine boundary.
    for key, scores in validation_scores.items():
        if len(scores) >= 5:
            REFERENCE_THRESHOLDS[key] = float(
                torch.tensor(scores).quantile(0.05).item()
            )

    print("Reference classes loaded:", list(REFERENCE_BANK.keys()))
    print("Provisional reference thresholds:", REFERENCE_THRESHOLDS)


def compare_with_reference(image, denomination):
    """Compare against available references for the predicted denomination."""

    key = str(denomination)
    references = REFERENCE_BANK.get(key)
    threshold = REFERENCE_THRESHOLDS.get(key)

    if references is None or threshold is None:
        return {
            "reference_similarity": None,
            "reference_screening": "Unavailable",
            "reference_message": (
                "Not enough reference data for this denomination. "
                "Authenticity remains unverified."
            ),
        }

    try:
        embedding = get_visual_embedding(image)

        similarity = float(
            (references @ embedding).max().item()
        )

        similarity_percent = round(
            max(0.0, min(1.0, similarity)) * 100,
            2,
        )

        if similarity < threshold:
            screening = "Low reference similarity — manual review"
            message = (
                "The image differs from available reference examples. "
                "Lighting, camera angle, folds, blur or background can "
                "also cause low similarity. This does not prove the note is fake."
            )
        else:
            screening = "Visually similar to reference examples"
            message = (
                "The image resembles at least one reference example. "
                "Visual similarity does not prove authenticity."
            )

        return {
            "reference_similarity": similarity_percent,
            "reference_screening": screening,
            "reference_message": message,
        }

    except (OSError, ValueError, RuntimeError) as error:
        print("Reference comparison error:", error)

        return {
            "reference_similarity": None,
            "reference_screening": "Unavailable",
            "reference_message": (
                "Reference comparison failed. Authenticity is unverified."
            ),
        }


# Build the reference bank once when the app starts.
# build_reference_bank()


# =========================================================
# IMAGE QUALITY CHECKS
# =========================================================

def check_image_quality(image):
    warnings = []
    width, height = image.size

    if min(width, height) < 250:
        warnings.append(
            "Image resolution is low. Try a higher-resolution photo."
        )

    gray = image.convert("L")
    brightness = ImageStat.Stat(gray).mean[0]

    if brightness < 45:
        warnings.append(
            "Image appears very dark. Improve lighting."
        )
    elif brightness > 225:
        warnings.append(
            "Image may be overexposed. Reduce glare."
        )

    edges = gray.filter(ImageFilter.FIND_EDGES)
    sharpness = ImageStat.Stat(edges).var[0]

    if sharpness < 45:
        warnings.append(
            "Image may be blurry. Hold the camera steady."
        )

    return warnings


# =========================================================
# EXPERIMENTAL AUTHENTICITY INFERENCE
# Dataset labels are unverified.
# =========================================================

def check_authenticity(image):
    if authenticity_model is None:
        return {
            "authenticity_status": "Unverified",
            "authenticity_message": (
                "Experimental authenticity model unavailable. "
                "The app cannot verify whether the note is genuine or counterfeit."
            ),
            "experimental_result": None,
        }

    try:
        tensor = transform(
            image.convert("RGB")
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            scores = torch.softmax(
                authenticity_model(tensor),
                dim=1,
            )[0]

        index = int(scores.argmax().item())
        predicted_class = authenticity_classes[index]
        score = float(scores[index].item() * 100)

        return {
            "authenticity_status": "Unverified",
            "authenticity_message": (
                "Experimental classification only. Training labels are "
                "unverified and cannot confirm authenticity."
            ),
            "experimental_result": {
                "category": predicted_class,
                "score": round(score, 2),
            },
        }

    except (RuntimeError, ValueError, IndexError) as error:
        print("Experimental inference error:", error)

        return {
            "authenticity_status": "Unverified",
            "authenticity_message": (
                "Experimental analysis failed. Authenticity was not verified."
            ),
            "experimental_result": None,
        }


# =========================================================
# MAIN FLASK ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("currency_image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select or capture a currency image."

        else:
            extension = Path(uploaded_file.filename).suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                error = "Please upload a JPG, JPEG, PNG, or WEBP image."

            else:
                saved_path = None

                try:
                    # Check that the uploaded file is a valid image.
                    with Image.open(uploaded_file.stream) as original:
                        original.verify()

                    uploaded_file.stream.seek(0)

                    with Image.open(uploaded_file.stream) as original:
                     original.thumbnail((1600, 1600))
                     image = original.convert("RGB")

                    quality_warnings = check_image_quality(image)

                    filename = f"{uuid.uuid4().hex}.jpg"
                    saved_path = UPLOAD_DIR / filename

                    image.save(
                        saved_path,
                        format="JPEG",
                        quality=92,
                    )

                    # 1. Denomination prediction.
                    tensor = transform(
                        image
                    ).unsqueeze(0).to(DEVICE)

                    with torch.inference_mode():
                        scores = torch.softmax(
                            model(tensor),
                            dim=1,
                        )[0]

                    confidence, predicted_index = scores.max(dim=0)

                    confidence_percent = float(
                        confidence.item() * 100
                    )

                    denomination = classes[
                        predicted_index.item()
                    ]

                    # 2. Experimental authenticity classification.
                    authenticity = check_authenticity(image)

                    # 3. Reference-based similarity screening.
                    reference_check = compare_with_reference(
                        image,
                        denomination,
                    )

                    result = {
                        "image_url": f"/static/uploads/{filename}",
                        "denomination": denomination,
                        "confidence": round(confidence_percent, 2),
                        "low_confidence": (
                            confidence_percent
                            < LOW_CONFIDENCE_THRESHOLD
                        ),
                        "quality_warnings": quality_warnings,
                        **authenticity,
                        **reference_check,
                    }

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    RuntimeError,
                ) as exception:
                    print("Image processing error:", exception)

                    error = (
                        "The image could not be processed. "
                        "Please choose a valid image and try again."
                    )

                    if saved_path and saved_path.exists():
                        saved_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


# =========================================================
# FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="Image is too large. Maximum upload size is 10 MB.",
    ), 413


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
