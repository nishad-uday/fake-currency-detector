
from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "raw"
    / "Indian Currency Dataset"
    / "Indian Currency Dataset"
)

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed_mendeley"

CLASSES = ["10", "20", "50", "100", "200", "500", "2000"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def main():
    if not SOURCE_DIR.exists():
        print(f"Dataset folder not found: {SOURCE_DIR}")
        return

    # Prevent accidental overwriting or duplicate copies.
    split_names = ["train", "val", "test"]
    if any((OUTPUT_DIR / name).exists() for name in split_names):
        print(f"Output split already exists: {OUTPUT_DIR}")
        print("Rename or move the existing processed split before running again.")
        return

    random.seed(SEED)
    summary = {name: 0 for name in split_names}
    total = 0

    for label in CLASSES:
        class_dir = SOURCE_DIR / label

        if not class_dir.exists():
            print(f"Missing denomination folder: {class_dir}")
            return

        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        if len(images) < 3:
            print(f"Not enough images in denomination {label}")
            return

        random.shuffle(images)

        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        # Ensure every split has at least one image.
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n - n_train - n_val

        if n_test < 1:
            print(f"Not enough images to split denomination {label}")
            return

        partitions = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, files in partitions.items():
            destination = OUTPUT_DIR / split / label
            destination.mkdir(parents=True, exist_ok=True)

            for index, image_path in enumerate(files):
                # Prefix the name to prevent collisions from nested folders.
                new_name = f"{index:05d}_{image_path.name}"
                shutil.copy2(image_path, destination / new_name)

            summary[split] += len(files)
            total += len(files)

        print(
            f"₹{label}: {len(partitions['train'])} train, "
            f"{len(partitions['val'])} val, "
            f"{len(partitions['test'])} test"
        )

    print("\nDataset split complete!")
    print(f"Training images:   {summary['train']}")
    print(f"Validation images: {summary['val']}")
    print(f"Test images:       {summary['test']}")
    print(f"Total copied:      {total}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("Original images have not been changed.")


if __name__ == "__main__":
    main()
