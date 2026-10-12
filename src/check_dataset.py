
from pathlib import Path
from PIL import Image, ImageStat
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Indian Currency Dataset"
    / "Indian Currency Dataset"
)
REPORT_DIR = PROJECT_ROOT / "dataset" / "reports"
REPORT_FILE = REPORT_DIR / "mendeley_quality_report.csv"

CLASSES = ["10", "20", "50", "100", "200", "500", "2000"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found:\n{DATASET_DIR}")
        print("Please check the dataset folder path.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for label in CLASSES:
        folder = DATASET_DIR / label
        if not folder.exists():
            print(f"Missing folder: {label}")
            continue

        files = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        print(f"\n₹{label}: {len(files)} images")

        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()

                with Image.open(path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    brightness = sum(ImageStat.Stat(image).mean) / 3

                flags = []
                if min(width, height) < 150:
                    flags.append("small_image")
                if brightness < 35:
                    flags.append("very_dark")

                rows.append({
                    "denomination": label,
                    "file": str(path.relative_to(DATASET_DIR)),
                    "width": width,
                    "height": height,
                    "brightness": round(brightness, 1),
                    "status": "REVIEW" if flags else "CHECK_VISUALLY",
                    "flags": ", ".join(flags)
                })

            except Exception as error:
                rows.append({
                    "denomination": label,
                    "file": str(path.relative_to(DATASET_DIR)),
                    "width": "",
                    "height": "",
                    "brightness": "",
                    "status": "CORRUPT",
                    "flags": str(error)
                })

    with REPORT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "denomination", "file", "width", "height",
                "brightness", "status", "flags"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal images checked: {len(rows)}")
    print(f"Report saved to:\n{REPORT_FILE}")
    print("Note: CHECK_VISUALLY does not mean an image is guaranteed clear.")

if __name__ == "__main__":
    main()
