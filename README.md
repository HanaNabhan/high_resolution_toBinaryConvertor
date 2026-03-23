# Grayscale to Binary Engraving Dithering

Convert grayscale animal photos into pure binary black-and-white images in engraving/stipple style.

## Result

| Input | Output |
|-------|--------|
| ![input](images/input_grey.jpg) | ![output](images/output_binary.png) |

---

## Approach

The pipeline has four stages, each solving a specific problem:

**1. CLAHE (Contrast Limited Adaptive Histogram Equalization)**
Local contrast enhancement using an 8×8 tile grid. Unlike global histogram equalization, CLAHE enhances contrast independently per region — this recovers fine fur texture and whisker detail in dark shadow areas without blowing out highlights.

**2. Unsharp Mask (Sharpening)**
A Gaussian blur is subtracted from the image with a strength of 1.6/−0.6. This accentuates edges before dithering so that fur strands and fine lines survive the binary quantization step as distinct marks rather than blending into background noise.

**3. Shadow Push (Non-linear LUT)**
A custom lookup table crushes pixels below brightness 130 toward black using a power curve `(i/130)^1.6 × 100`. Pixels at or above 130 pass through completely unchanged. This causes dark regions (jaw, eye shadow, dense fur) to produce tight dense clusters in the dithered output — the hallmark of engraving style — while mid-tones and highlights are unaffected.

**4. JJN Error Diffusion Dithering**
Jarvis-Judice-Ninke dithering spreads quantization error across 12 neighboring pixels in a wide fan pattern (2 rows deep, 5 columns wide). Compared to simpler algorithms like Floyd-Steinberg (7 neighbors), JJN produces finer, more organic dot texture that reads as stipple engraving rather than mechanical halftone. Each pixel is thresholded at 128 and the rounding error propagates forward — meaning every tone in the source is faithfully represented as a locally correct dot density in the output.

---

## Usage

```
pip install opencv-python pillow numpy
```

Place your input image at `images/input_grey.jpg`, then run:

```
python scripts/run.py
```

Output is saved to `images/output_binary.jpg`.

The script auto-resizes any image wider than 1200px while preserving aspect ratio.

---

## Project Structure

```
├── images/
│   ├── input_grey.jpg       ← grayscale input photo
│   └── output_binary.jpg    ← pure binary output
└── scripts/
    └── run.py               ← full pipeline
```
