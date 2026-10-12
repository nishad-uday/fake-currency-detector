
import csv
from pathlib import Path
from collections import Counter

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "authenticity_final"
MANIFEST_PATH = DATASET_ROOT / "reports" / "dataset_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    print("=" * 60)
    print("FAKE CURRENCY DATASET VALIDATION")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset folder not found:\n{DATASET_ROOT}")
        return

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found:\n{MANIFEST_PATH}")
        return

    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    print(f"\nManifest entries: {len(rows)}")
    print(f"Columns: {columns}")

    missing = []
    existing = 0
    label_counts = Counter()
    denomination_counts = Counter()
    image_paths = set()

    for row in rows:
        relative_path = row.get("image_path", "").strip()
        if not relative_path:
            missing.append("(empty image_path)")
            continue

        image_path = DATASET_ROOT / relative_path
        image_paths.add(relative_path)

        if image_path.is_file():
            existing += 1
        else:
            missing.append(relative_path)

        label = row.get("verification_use_for_training", "").strip()
        if not label:
            label = row.get("label", "").strip() or "missing_label"
        label_counts[label] += 1

        denomination = row.get("denomination", "").strip() or "unknown"
        denomination_counts[denomination] += 1

    actual_images = [
        path for path in DATASET_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Image files found: {len(actual_images)}")
    print(f"Manifest paths: {len(image_paths)}")
    print(f"Manifest paths existing: {existing}")
    print(f"Missing image paths: {len(missing)}")

    print("\n--- Denomination counts in manifest ---")
    for name, count in sorted(denomination_counts.items()):
        print(f"{name}: {count}")

    print("\n--- Label counts in manifest ---")
    for name, count in label_counts.most_common():
        print(f"{name}: {count}")

    if missing:
        print("\n--- First 20 missing paths ---")
        for path in missing[:20]:
            print(path)

    print("\n--- Important notes ---")
    print("Manifest labels are claims, not proof of genuine/fake status.")
    print("Feature crops are not equivalent to full banknote images.")
    print("Do not train the authenticity model until labels are reviewed.")
    print("\nValidation finished.")

if __name__ == "__main__":
    main()
