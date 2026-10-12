
from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "dataset" / "note_candidates"
OUTPUT_PATH = INPUT_DIR / "candidate_contact_sheet.jpg"

files = sorted(INPUT_DIR.glob("*_candidate_*.jpg"))

if not files:
    raise FileNotFoundError(f"No candidate crops found in {INPUT_DIR}")

columns = 4
thumb_w, thumb_h = 240, 150
label_h = 35
rows = math.ceil(len(files) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 12))

    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    sheet.paste(image, (x + 6, y + 6))
    draw.text(
        (x + 6, y + thumb_h),
        path.name[:32],
        fill="black"
    )

sheet.save(OUTPUT_PATH, quality=90)

print(f"Candidate images reviewed: {len(files)}")
print(f"Contact sheet saved: {OUTPUT_PATH}")
