
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "source_classifier_report.json"

if not REPORT.exists():
    raise SystemExit("Report not found. Train the source classifier first.")

data = json.loads(REPORT.read_text(encoding="utf-8"))

print("=" * 45)
print("SOURCE CLASSIFIER EVALUATION SUMMARY")
print("=" * 45)
print("Purpose:", data["purpose"])
print("Classes:", ", ".join(data["classes"]))
print("Total images:", data["images"])
print("Training images:", data["train_images"])
print("Validation images:", data["validation_images"])
print("Validation accuracy:", data["last_validation_accuracy_percent"], "%")

print("\nLimitations:")
for item in data["limitations"]:
    print("-", item)

print("\nThis model does NOT identify genuine or counterfeit currency.")
