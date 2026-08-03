#!/usr/bin/env python3
"""
COWC Degraded Test Set Generator
--------------------------------
1. Extracts 512x512 patches from raw COWC test set cities using sliding window (stride 256).
2. Converts COWC annotation mask points to local patch pixel centerpoints (x y).
3. Applies optical lens blur and electronic sensor noise (degradation).
4. Saves directly into the format expected by evaluate_detr_rq1.py.
"""

import os
import glob
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
RAW_COWC_DIR = "data/COWC"  # Source directory containing raw COWC city folders

# Output directory matching evaluate_detr_rq1.py structure: test_data/COWC/<variant>/
OUTPUT_DIR = "test_data/COWC/15cm_sensor_degraded"

WINDOW_SIZE = 512  # Patch dimension (pixels)
STRIDE = 256       # 50% overlap

# Cities designated for the test split
TEST_CITIES = ["Potsdam_ISPRS", "Columbus_CSUAV_AFRL"]

# Sensor degradation parameters
BLUR_KERNEL = (3, 3)
NOISE_SIGMA = 12.0
APPLY_DEGRADATION = True  # Set to False if you want clean patches instead


# ---------------------------------------------------------
# SENSOR DEGRADATION FUNCTION
# ---------------------------------------------------------
def apply_sensor_degradations(image):
    """
    Applies optical blur (Gaussian) and electronic thermal/shot noise (Gaussian).
    """
    # 1. Optical Lens Blur
    img_blurred = cv2.GaussianBlur(image, BLUR_KERNEL, 0)

    # 2. Electronic Sensor Noise
    row, col, ch = img_blurred.shape
    gauss_noise = np.random.normal(0, NOISE_SIGMA, (row, col, ch)).astype(np.float32)

    # Add noise & clamp pixel bounds [0, 255]
    noisy_img = img_blurred.astype(np.float32) + gauss_noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def setup_directories(output_root):
    img_dir = output_root / "images"
    lbl_dir = output_root / "labels"

    if output_root.exists():
        print(f"🧹 Clearing existing directory at '{output_root}'...")
        shutil.rmtree(output_root)

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def process_city_images(city_dir):
    """Finds matching image and annotation pairs in a city folder."""
    all_pngs = glob.glob(os.path.join(RAW_COWC_DIR, city_dir, "*.png"))
    pairs = []

    for img_path in all_pngs:
        if "_Annotated_" in img_path:
            continue

        base, ext = os.path.splitext(img_path)
        anno_path = f"{base}_Annotated_Cars.png"

        if os.path.exists(anno_path):
            pairs.append((img_path, anno_path))

    return pairs


# ---------------------------------------------------------
# MAIN PATCH EXTRACTION & DEGRADATION LOOP
# ---------------------------------------------------------
def extract_and_degrade_patches(out_img_dir, out_lbl_dir):
    print("\n=== Generating Degraded COWC Test Dataset ===")
    print(f" Source Data: {RAW_COWC_DIR}")
    print(f" Output Dir : {out_img_dir.parent}")
    print(f" Patch Size : {WINDOW_SIZE}x{WINDOW_SIZE} | Stride: {STRIDE}")
    print(f" Degradation: Gaussian Blur {BLUR_KERNEL} + Noise Sigma={NOISE_SIGMA}\n")

    total_patch_count = 0

    for city in TEST_CITIES:
        city_path = os.path.join(RAW_COWC_DIR, city)
        if not os.path.exists(city_path):
            print(f"⚠️  Warning: City folder '{city}' not found in {RAW_COWC_DIR}. Skipping.")
            continue

        pairs = process_city_images(city)
        print(f"--> Processing {city} ({len(pairs)} large-scale imagery pairs)...")

        for img_path, anno_path in tqdm(pairs, desc=f"    Extracting & Degrading {city}"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            anno = cv2.imread(anno_path)
            anno_gray = cv2.cvtColor(anno, cv2.COLOR_BGR2GRAY)
            _, pts_mask = cv2.threshold(anno_gray, 1, 255, cv2.THRESH_BINARY)

            h, w, _ = img.shape
            base_filename = Path(img_path).stem

            # Sliding window over the full image
            for y in range(0, h - WINDOW_SIZE + 1, STRIDE):
                for x in range(0, w - WINDOW_SIZE + 1, STRIDE):

                    patch_mask = pts_mask[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]
                    y_indices, x_indices = np.where(patch_mask > 0)

                    # Skip empty patches with no vehicles
                    if len(x_indices) == 0:
                        continue

                    # Extract the raw image patch
                    patch_img = img[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE]

                    # 1. Apply Sensor Physics Degradation
                    if APPLY_DEGRADATION:
                        processed_img = apply_sensor_degradations(patch_img)
                    else:
                        processed_img = patch_img

                    # 2. Save Image
                    file_stem = f"{city}_{base_filename}_patch_{y}_{x}"
                    img_out_path = out_img_dir / f"{file_stem}.png"
                    cv2.imwrite(str(img_out_path), processed_img)

                    # 3. Save Ground Truth COWC Pixel Centerpoints (x y)
                    label_out_path = out_lbl_dir / f"{file_stem}.txt"
                    with open(label_out_path, "w") as f:
                        for px, py in zip(x_indices, y_indices):
                            # Save as relative pixel coordinates inside the 512x512 patch
                            # Class ID 0 included to comply with evaluate_detr_rq1.py parsing
                            f.write(f"0 {float(px):.2f} {float(py):.2f}\n")

                    total_patch_count += 1

    print(f"\n✔ Done! Successfully created {total_patch_count} degraded test patches.")


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == "__main__":
    out_root = Path(OUTPUT_DIR)
    out_img_dir, out_lbl_dir = setup_directories(out_root)
    extract_and_degrade_patches(out_img_dir, out_lbl_dir)