
from pathlib import Path
import sys

import cv2
import numpy as np

def analyze_image(image_path):
    path = Path(image_path)

    if not path.is_file():
        print("ERROR: Image file not found:", path)
        return

    image = cv2.imread(str(path))

    if image is None:
        print("ERROR: Image could not be read.")
        return

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    print("\n===== IMAGE QUALITY REPORT =====")
    print(f"Image: {path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"Brightness: {brightness:.2f}/255")
    print(f"Sharpness score: {sharpness:.2f}")

    warnings = []

    if min(width, height) < 250:
        warnings.append("Low resolution")

    if brightness < 60:
        warnings.append("Image is dark")
    elif brightness > 220:
        warnings.append("Image may be overexposed")

    if sharpness < 50:
        warnings.append("Image may be blurry")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print("-", warning)
    else:
        print("\nNo basic image-quality warning detected.")

    print("\nNote: Quality scores do not establish genuine/fake status.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python src/test_image_quality.py "path/to/image.jpg"')
    else:
        analyze_image(sys.argv[1])
