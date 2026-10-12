
from pathlib import Path
from PIL import Image, ImageDraw
import math

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "dataset" / "pdf_extracted"
OUTPUT_PATH = INPUT_DIR / "pdf_contact_sheet.jpg"

THUMB_WIDTH = 240
THUMB_HEIGHT = 180
LABEL_HEIGHT = 30
COLUMNS = 4

image_files = sorted(INPUT_DIR.glob("pdf_page_*.jpg"))

if not image_files:
    raise FileNotFoundError(
        f"No PDF page images found in: {INPUT_DIR}"
    )

rows = math.ceil(len(image_files) / COLUMNS)
cell_height = THUMB_HEIGHT + LABEL_HEIGHT

sheet = Image.new(
    "RGB",
    (COLUMNS * THUMB_WIDTH, rows * cell_height),
    "white"
)
draw = ImageDraw.Draw(sheet)

for index, image_path in enumerate(image_files):
    with Image.open(image_path) as original:
        img = original.convert("RGB")
        img.thumbnail((THUMB_WIDTH - 10, THUMB_HEIGHT - 10))

    col = index % COLUMNS
    row = index // COLUMNS

    x = col * THUMB_WIDTH
    y = row * cell_height

    sheet.paste(img, (x + 5, y + 5))
    draw.text(
        (x + 5, y + THUMB_HEIGHT + 3),
        image_path.stem,
        fill="black"
    )

sheet.save(OUTPUT_PATH, quality=90)

print(f"Pages included: {len(image_files)}")
print(f"Contact sheet saved: {OUTPUT_PATH}")
