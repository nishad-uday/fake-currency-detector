
from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "currency_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "path/to/image.jpg"')
        return

    image_path = Path(sys.argv[1])

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return

    if not MODEL_PATH.is_file():
        print(f"Model not found: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    # Re-create the same ResNet18 architecture used during training.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    top_values, top_indices = torch.topk(
        probabilities, k=min(3, len(classes))
    )

    print("\nTop 3 predictions:")
    for probability, index in zip(top_values, top_indices):
        print(
            f"₹{classes[index.item()]}: "
            f"{probability.item():.2%}"
        )

    print("\nThese are model scores, not verified probabilities.")
    print("This model predicts denomination, not genuine/fake status.")


if __name__ == "__main__":
    main()
