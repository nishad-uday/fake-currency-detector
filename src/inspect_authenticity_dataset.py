
from pathlib import Path
from collections import Counter, defaultdict
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "authenticity_ready"
OUTPUT = ROOT / "dataset" / "authenticity_inventory.csv"

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
files = [
    p for p in DATA.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

rows = []
source_counts = Counter()
denomination_counts = Counter()
source_denomination_counts = Counter()

for path in files:
    relative = path.relative_to(DATA)
    parts = relative.parts
    source = parts[0] if parts else "unknown"

    denomination = "unknown"
    for part in parts[1:]:
        if part in {"10", "20", "50", "100", "200", "500", "2000"}:
            denomination = part
            break

    if denomination == "unknown":
        prefix = path.stem.split("_")[0]
        if prefix in {"10", "20", "50", "100", "200", "500", "2000"}:
            denomination = prefix

    source_counts[source] += 1
    denomination_counts[denomination] += 1
    source_denomination_counts[(source, denomination)] += 1

    rows.append({
        "filename": path.name,
        "relative_path": str(relative),
        "source_folder": source,
        "denomination_guess": denomination,
        "authenticity_label": "UNVERIFIED"
    })

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Total images:", len(files))
print("\nSource folders:")
for name, count in sorted(source_counts.items()):
    print(f"  {name}: {count}")

print("\nDenomination guesses (from folder/filename only):")
for name, count in sorted(denomination_counts.items()):
    print(f"  ₹{name}: {count}" if name != "unknown" else f"  Unknown: {count}")

print("\nSource-wise denomination:")
for (source, denomination), count in sorted(source_denomination_counts.items()):
    print(f"  {source} | ₹{denomination}: {count}")

print("\nInventory saved to:", OUTPUT)
print("All authenticity labels remain UNVERIFIED.")
