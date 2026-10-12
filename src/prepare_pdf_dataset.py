
from pathlib import Path
import csv
import fitz

# PDF ka path: apne PC par is location ko verify karna
# PDF_PATH = Path(r"D:\temp project\orignal note.pdf")


PDF_PATH = Path(r"D:\temp project\fake-currency-detector\orignal note.pdf")


# Project root folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Output folder
OUTPUT_DIR = PROJECT_DIR / "dataset" / "pdf_extracted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "currency_dataset.csv"


def find_pdf():
    """Find PDF at the configured path or common project locations."""
    possible_paths = [
        PDF_PATH,
        PROJECT_DIR / "orignal note.pdf",
        PROJECT_DIR / "original note.pdf",
        PROJECT_DIR / "orignal_note.pdf",
        PROJECT_DIR / "original_note.pdf",
    ]

    for path in possible_paths:
        if path.is_file():
            print(f"PDF found: {path}")
            return path

    checked = "\n".join(str(path) for path in possible_paths)
    raise FileNotFoundError(
        "PDF nahi mili. PDF ko project root folder mein copy karo:\n"
        f"{PROJECT_DIR}\n\n"
        "Checked locations:\n" + checked
    )


def main():
    pdf_path = find_pdf()
    rows = []

    with fitz.open(pdf_path) as pdf:
        total_pages = len(pdf)
        print(f"Total PDF pages: {total_pages}")

        for page_index, page in enumerate(pdf):
            page_no = page_index + 1

            image_name = f"pdf_page_{page_no:03d}.jpg"
            image_path = OUTPUT_DIR / image_name

            # Render page at higher resolution
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2, 2),
                alpha=False
            )
            pix.save(str(image_path))

            rows.append({
                "filename": image_name,
                "image_path": str(image_path.relative_to(PROJECT_DIR)),
                "page_number": page_no,
                "denomination": "needs_review",
                "side": "needs_review",
                "authenticity_label": "unverified",
                "source": "user_supplied_pdf",
                "review_status": "pending",
            })

            print(f"Exported {page_no}/{total_pages}: {image_name}")

    if rows:
        with CSV_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\nPDF extraction completed!")
    print(f"Pages exported: {len(rows)}")
    print(f"Images folder: {OUTPUT_DIR}")
    print(f"Dataset sheet: {CSV_PATH}")


if __name__ == "__main__":
    main()
