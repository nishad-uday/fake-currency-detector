
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "authenticity_ready" / "repository_fake_unverified"
OUTPUT = ROOT / "dataset" / "claimed_fake_contact_sheet.jpg"

files = sorted(
    p for p in SOURCE.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
)

if not files:
    raise SystemExit(f"No images found in {SOURCE}")

cell_w, cell_h = 300, 210
label_h = 32
columns = 3
rows = math.ceil(len(files) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_w, rows * (cell_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    with Image.open(path) as im:
        thumb = ImageOps.contain(im.convert("RGB"), (cell_w - 12, cell_h - 12))

    x = (i % columns) * cell_w
    y = (i // columns) * (cell_h + label_h)

    sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
    draw.text((x + 8, y + cell_h + 5), path.name, fill="black")

sheet.save(OUTPUT, quality=90)
print("Images:", len(files))
print("Contact sheet saved to:", OUTPUT)
print("Folder labels are unverified; this sheet does not establish authenticity.")
