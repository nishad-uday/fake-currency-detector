
from pathlib import Path
import csv

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "authenticity_ready"
OUTPUT = ROOT / "dataset" / "authenticity_image_analysis.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
rows = []

for path in sorted(DATA.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    image = cv2.imread(str(path))
    if image is None:
        continue

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    source = path.relative_to(DATA).parts[0]

    rows.append({
        "filename": path.name,
        "relative_path": str(path.relative_to(DATA)),
        "source": source,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "authenticity_label": "UNVERIFIED",
    })

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Images analyzed:", len(rows))
print("CSV saved:", OUTPUT)
print("Authenticity labels remain UNVERIFIED.")
