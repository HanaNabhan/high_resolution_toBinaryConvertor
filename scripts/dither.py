"""
High-Detail Binary Engraving Dithering Script
Pipeline: Grayscale → CLAHE → Tone Curve → Sharpen → Jarvis Dithering → 1-bit PNG
"""

import cv2
import numpy as np
from PIL import Image
import os

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_tone_curve(img):
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        y = x ** 1.3
        lut[i] = np.clip(int(y * 255), 0, 255)
    return cv2.LUT(img, lut)

def apply_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
    sharpened = cv2.addWeighted(img, 1.6, blurred, -0.6, 0)
    return sharpened

def jarvis_judice_ninke_dither(img_array):
    h, w = img_array.shape
    img = img_array.astype(np.float32)
    output = np.zeros((h, w), dtype=np.uint8)

    offsets = [
        (0, 1, 7), (0, 2, 5),
        (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
        (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1),
    ]

    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            output[y, x] = new_pixel
            error = old_pixel - new_pixel
            for dy, dx, weight in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += error * weight / 48.0

        if y % 50 == 0:
            print(f"  Row {y}/{h}", end='\r')

    print()
    return output

def process_image(input_path, output_path):
    print(f"\nProcessing: {os.path.basename(input_path)}")

    img = cv2.imread(input_path)
    if img is None:
        print(f"ERROR loading {input_path}")
        return

    # Resize for faster processing while keeping quality
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        print(f"  Resized to {img.shape[1]}x{img.shape[0]}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = apply_clahe(gray)
    toned = apply_tone_curve(enhanced)
    sharpened = apply_sharpen(toned)

    print("  Running JJN dithering...")
    dithered = jarvis_judice_ninke_dither(sharpened)

    pil_img = Image.fromarray(dithered).convert('1')
    pil_img.save(output_path, format='PNG')

    verify = np.unique(np.array(Image.open(output_path).convert('L')))
    print(f"  ✓ Saved → pixel values: {verify}")

os.makedirs("/mnt/user-data/outputs", exist_ok=True)

files = [
    ("tiger1.jpg", "../output/tiger1.png"),
    # ("/mnt/user-data/uploads/5165439cd78c05bbf59dc64d569e9b33.jpg",        "/mnt/user-data/outputs/tiger2_engraving.png"),
    # ("/mnt/user-data/uploads/lion_binary.jpg",                              "/mnt/user-data/outputs/lion_engraving.png"),
]

for inp, out in files:
    process_image(inp, out)

print("\n✅ Done!")
