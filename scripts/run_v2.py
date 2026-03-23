import cv2
import numpy as np
from PIL import Image
import os
 
def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)
 
def apply_scurve(img):
    """
    Strong S-curve tone mapping:
      - Shadows (0-90):    compressed hard toward black → dense clusters
      - Midtones (90-166): gently stretched for contrast → fur detail preserved
      - Highlights (166+): pushed toward white → sparse dots in bright areas
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        if x < 0.35:
            y = x * 0.4
        elif x < 0.65:
            y = 0.14 + (x - 0.35) * (0.72 / 0.30)
        else:
            y = 0.86 + (x - 0.65) * (0.14 / 0.35)
        lut[i] = np.clip(int(y * 255), 0, 255)
    return cv2.LUT(img, lut)
 
def apply_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
 
def jjn_dither(img_array):
    """
    Jarvis-Judice-Ninke error diffusion dithering.
    Spreads quantization error to 12 neighboring pixels,
    producing fine organic dot texture that preserves tonal detail.
    """
    h, w = img_array.shape
    img = img_array.astype(np.float32)
    output = np.zeros((h, w), dtype=np.uint8)
    offsets = [
        (0,1,7),(0,2,5),
        (1,-2,3),(1,-1,5),(1,0,7),(1,1,5),(1,2,3),
        (2,-2,1),(2,-1,3),(2,0,5),(2,1,3),(2,2,1),
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
            print(f"  Dithering row {y}/{h}", end='\r')
    print()
    return output
 
def process_image(input_path, output_path):
    print(f"Loading: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {input_path}")
 
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    print(f"  Size: {img.shape[1]}x{img.shape[0]}")
 
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced  = apply_clahe(gray)
    curved    = apply_scurve(enhanced)
    sharpened = apply_sharpen(curved)
 
    print("  Running JJN dithering...")
    dithered = jjn_dither(sharpened)
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(dithered).convert('1').save(output_path, format='PNG')
 
    black_pct = (np.array(Image.open(output_path).convert('L')) == 0).mean() * 100
    print(f"  ✓ Saved → {output_path}  |  black density: {black_pct:.1f}%")
 
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
process_image(
    os.path.join(BASE, "images_v3", "input_grey.jpg"),
    os.path.join(BASE, "images_v3", "output_binary.jpg"),
)
 
print("\n Done!")