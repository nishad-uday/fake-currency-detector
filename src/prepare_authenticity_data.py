
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "authenticity_final"
OUTPUT = ROOT / "dataset" / "authenticity_ready"

# Sirf full-note images wale folders.
# Feature crops ko jaanbujhkar include nahi kiya gaya.
SOURCES = {
    "user_photos_unverified": DATA / "authenticity" / "genuine_user",
    "repository_fake_unverified": (
        DATA / "authenticity" / "counterfeit_github_unverified"
    ),
    "repository_reference_unverified": (
        DATA / "reference_genuine_github"
    ),
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main():
    if not DATA.exists():
        print("ERROR: Dataset folder nahi mila:", DATA)
        return

    total = 0

    for category, source in SOURCES.items():
        if not source.exists():
            print(f"SKIP (folder nahi mila): {source}")
            continue

        images = [
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]

        destination = OUTPUT / category
        destination.mkdir(parents=True, exist_ok=True)

        copied = 0
        for image in images:
            target = destination / image.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            copied += 1

        total += copied
        print(f"{category}: {copied} images copied")

    print("\nTotal copied:", total)
    print("Output:", OUTPUT)
    print("IMPORTANT: Ye images abhi unverified hain.")
    print("Inhe abhi model training mein use mat karna.")

if __name__ == "__main__":
    main()
