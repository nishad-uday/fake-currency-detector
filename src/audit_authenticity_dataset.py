
from pathlib import Path
from collections import defaultdict
import csv
import hashlib

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "authenticity_ready"
REPORT = ROOT / "dataset" / "authenticity_audit.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
hash_groups = defaultdict(list)
rows = []

for path in sorted(DATASET.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    relative_path = str(path.relative_to(ROOT))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hash_groups[digest].append(relative_path)

    image = cv2.imread(str(path))
    if image is None:
        rows.append({
            "file": relative_path,
            "width": 0,
            "height": 0,
            "brightness": "",
            "sharpness": "",
            "status": "UNREADABLE",
            "duplicate_group": digest[:12],
        })
        continue

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    if min(width, height) < 250:
        warnings.append("LOW_RESOLUTION")
    if brightness < 60:
        warnings.append("DARK")
    elif brightness > 220:
        warnings.append("OVEREXPOSED")
    if sharpness < 50:
        warnings.append("POSSIBLY_BLURRY")

    rows.append({
        "file": relative_path,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "status": ";".join(warnings) if warnings else "NO_BASIC_WARNING",
        "duplicate_group": digest[:12],
    })

REPORT.parent.mkdir(parents=True, exist_ok=True)

with REPORT.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

duplicates = [group for group in hash_groups.values() if len(group) > 1]
warning_count = sum(
    row["status"] not in {"NO_BASIC_WARNING"}
    for row in rows
)

print("\n===== DATASET AUDIT REPORT =====")
print("Images scanned:", len(rows))
print("Unreadable images:", sum(r["status"] == "UNREADABLE" for r in rows))
print("Images with quality warnings:", warning_count)
print("Exact duplicate groups:", len(duplicates))
print("CSV report:", REPORT)

if duplicates:
    print("\nExact duplicates:")
    for group in duplicates:
        print(" -", " | ".join(group))

print("\nAudit complete. No images were moved or deleted.")
print("Note: Image quality and duplicate checks do not verify authenticity.")
