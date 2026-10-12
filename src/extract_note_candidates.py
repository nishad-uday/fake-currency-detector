
from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "dataset" / "pdf_extracted"
OUTPUT_DIR = ROOT / "dataset" / "note_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "candidates.csv"

MIN_AREA = 1500
MIN_WIDTH = 35
MIN_HEIGHT = 20

rows = []
count = 0

for page_path in sorted(INPUT_DIR.glob("pdf_page_*.jpg")):
    image = cv2.imread(str(page_path))

    if image is None:
        print(f"Skipped unreadable page: {page_path.name}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Separate darker printed regions from the light page background.
    _, binary = cv2.threshold(
        gray, 235, 255, cv2.THRESH_BINARY_INV
    )

    # Join nearby printed details into larger candidate regions.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (7, 7)
    )
    joined = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < MIN_AREA or w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        # Ignore regions occupying almost the entire page.
        if w > image.shape[1] * 0.95 and h > image.shape[0] * 0.95:
            continue

        candidates.append((x, y, w, h))

    # Stable top-to-bottom, left-to-right ordering.
    candidates.sort(key=lambda b: (b[1], b[0]))

    for index, (x, y, w, h) in enumerate(candidates, start=1):
        # Add margin so note edges are less likely to be cut off.
        margin = 12
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        crop = image[y1:y2, x1:x2]
        count += 1

        filename = (
            f"{page_path.stem}_candidate_{index:02d}.jpg"
        )
        output_path = OUTPUT_DIR / filename

        cv2.imwrite(str(output_path), crop)

        rows.append({
            "filename": filename,
            "source_page": page_path.name,
            "candidate_number": index,
            "denomination": "needs_review",
            "side": "needs_review",
            "authenticity_label": "unverified",
            "review_status": "pending",
            "crop_x": x1,
            "crop_y": y1,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
        })

    print(
        f"{page_path.name}: "
        f"{len(candidates)} candidate regions"
    )

if rows:
    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

print("\nCandidate extraction finished.")
print(f"Candidate crops: {count}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"CSV manifest: {CSV_PATH}")
print("All denomination and authenticity labels need review.")
