
from pathlib import Path
from PIL import Image
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset" / "authenticity_ready"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    if not DATASET.exists():
        print("Dataset folder not found:", DATASET)
        return

    total = 0
    valid = 0
    corrupt = []
    sizes = Counter()

    for folder in sorted(DATASET.iterdir()):
        if not folder.is_dir():
            continue

        images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        folder_valid = 0

        for path in images:
            total += 1
            try:
                with Image.open(path) as img:
                    img.verify()

                with Image.open(path) as img:
                    sizes[img.size] += 1

                valid += 1
                folder_valid += 1

            except Exception as error:
                corrupt.append((str(path), str(error)))

        print(f"{folder.name}: {folder_valid}/{len(images)} readable")

    print("\n--- SUMMARY ---")
    print("Total images:", total)
    print("Readable images:", valid)
    print("Corrupt/unreadable images:", len(corrupt))

    print("\nMost common image dimensions:")
    for size, count in sizes.most_common(10):
        print(f"{size[0]} x {size[1]}: {count}")

    if corrupt:
        print("\nFirst 10 unreadable files:")
        for path, error in corrupt[:10]:
            print(path, "->", error)

    print("\nCheck complete. No files were changed.")
    print("Note: readable images are not automatically genuine or fake.")

if __name__ == "__main__":
    main()
