
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "authenticity_ready"
OUTPUT = ROOT / "dataset" / "reference_comparison.jpg"

GROUPS = [
    ("Repository claimed fake", DATA / "repository_fake_unverified"),
    ("Repository reference", DATA / "repository_reference_unverified"),
]

THUMB = (280, 170)
CELL_W, CELL_H = 300, 210
COLS = 3
items = []

for group_name, folder in GROUPS:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }:
            items.append((group_name, path))

rows = math.ceil(len(items) / COLS)
sheet = Image.new("RGB", (COLS * CELL_W, rows * CELL_H), "white")
draw = ImageDraw.Draw(sheet)

for i, (group_name, path) in enumerate(items):
    with Image.open(path) as image:
        thumb = ImageOps.contain(image.convert("RGB"), THUMB)

    x = (i % COLS) * CELL_W
    y = (i // COLS) * CELL_H

    sheet.paste(thumb, (x + 10, y + 5))
    draw.text((x + 10, y + 178), group_name, fill="black")
    draw.text((x + 10, y + 193), path.name, fill="black")

sheet.save(OUTPUT, quality=90)
print("Images compared:", len(items))
print("Report saved:", OUTPUT)
print("Group labels are unverified; visual similarity is not authentication.")
