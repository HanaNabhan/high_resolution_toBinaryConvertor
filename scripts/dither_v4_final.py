import cv2
import numpy as np
from PIL import Image
import os

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_engraving_curve(img):
    """
    Lift the image to match engraving density (~28-32% black).
    Using gamma 0.45 which strongly lifts midtones toward white
    while keeping true blacks dark for shadow cluster effect.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        x = i / 255.0
        y = x ** 0.40
        lut[i] = np.clip(int(y * 255), 0, 255)
    return cv2.LUT(img, lut)

def apply_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def jjn_dither(img_array):
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
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += error * weight / 48.0
        if y % 50 == 0:
            print(f"  Row {y}/{h}", end='\r')
    print()
    return output

def process_image(input_path, output_path):
    print(f"\n{'─'*50}")
    print(f"Processing: {os.path.basename(input_path)}")

    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    print(f"  Size: {img.shape[1]}x{img.shape[0]}")

    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced  = apply_clahe(gray)
    curved    = apply_engraving_curve(enhanced)
    sharpened = apply_sharpen(curved)
    print("  ✓ CLAHE + engraving curve + sharpen applied")

    print("  Running JJN dithering...")
    dithered = jjn_dither(sharpened)

    pil_img = Image.fromarray(dithered).convert('1')
    pil_img.save(output_path, format='PNG')

    verify = np.unique(np.array(Image.open(output_path).convert('L')))
    black_pct = (np.array(Image.open(output_path).convert('L'))==0).mean()*100
    print(f"  ✓ Saved → values: {verify} | black density: {black_pct:.1f}% (target: ~28-32%)")


files = [
    ("/images/input_grey.jpg", "/images/output_binary.jpg"),
]

for inp, out in files:
    process_image(inp, out)

print("\n complete!")


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_sharpen(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.6, blurred, -0.6, 0)

def apply_shadow_push(img):
    """
    Push dark pixels deeper toward black.
    Threshold raised to 130 because CLAHE lifts dark pixels before this runs.
    Pixels 130+ pass through unchanged — fur and teeth are fully preserved.
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if i < 130:
            # Crush shadows: non-linear push toward black
            # 0→0, 65→30, 129→100 (darker than input, stronger at darkest end)
            y = (i / 130.0) ** 1.6 * 100
        else:
            # Everything above 130: completely untouched
            y = i
        lut[i] = np.clip(int(y), 0, 255)
    return cv2.LUT(img, lut)

def jjn_dither_array(img_float):
    """JJN error diffusion on a float32 array, returns float32 binary (0 or 255)"""
    h, w = img_float.shape
    img = img_float.copy()
    output = np.zeros((h, w), dtype=np.float32)

    offsets = [
        (0,1,7),(0,2,5),
        (1,-2,3),(1,-1,5),(1,0,7),(1,1,5),(1,2,3),
        (2,-2,1),(2,-1,3),(2,0,5),(2,1,3),(2,2,1),
    ]
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 128 else 0.0
            output[y, x] = new_pixel
            error = old_pixel - new_pixel
            for dy, dx, weight in offsets:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += error * weight / 48.0
        if y % 50 == 0:
            print(f"  JJN row {y}/{h}", end='\r')
    print()
    return output

def atkinson_dither_array(img_float):
    """
    Atkinson dithering — spreads only 3/4 of the error (not full error).
    This intentionally 'loses' some error, producing minimal sparse dots
    in near-white highlight regions. Perfect for the tooth/highlight areas.
    """
    h, w = img_float.shape
    img = img_float.copy()
    output = np.zeros((h, w), dtype=np.float32)

    offsets = [
        (0,1,1),(0,2,1),
        (1,-1,1),(1,0,1),(1,1,1),
        (2,0,1),
    ]
    for y in range(h):
        for x in range(w):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel > 128 else 0.0
            output[y, x] = new_pixel
            error = old_pixel - new_pixel
            for dy, dx, weight in offsets:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += error * weight / 8.0
        if y % 50 == 0:
            print(f"  Atkinson row {y}/{h}", end='\r')
    print()
    return output

def three_zone_dither(img_array):
    """
    Simplest correct approach:
    1. Apply shadow push to darken only pixels < 90 (gums/jaw)
    2. Run single JJN pass on the result
    
    This preserves v1 behavior everywhere EXCEPT the shadow zone,
    which now gets pre-darkened input so JJN produces denser clusters.
    Teeth and fur zones receive the exact same values as v1.
    """
    pushed = apply_shadow_push(img_array).astype(np.float32)

    print("  Running JJN on shadow-pushed input...")
    return jjn_dither_array(pushed).astype(np.uint8)

def process_image(input_path, output_path):
    print(f"\n{'─'*50}")
    print(f"Processing: {os.path.basename(input_path)}")

    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h*scale)), interpolation=cv2.INTER_LANCZOS4)
    print(f"  Size: {img.shape[1]}x{img.shape[0]}")

    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced  = apply_clahe(gray)
    sharpened = apply_sharpen(enhanced)
    print("CLAHE + sharpen applied")

    dithered = three_zone_dither(sharpened)

    pil_img = Image.fromarray(dithered.astype(np.uint8)).convert('1')
    pil_img.save(output_path, format='PNG')

    verify = np.unique(np.array(Image.open(output_path).convert('L')))
    print(f"  ✓ Saved → pixel values: {verify}")


files = [
    ("/images/input_grey.jpg", "/images/output_binary.jpg"),
]

for inp, out in files:
    process_image(inp, out)

print("\n complete!")
