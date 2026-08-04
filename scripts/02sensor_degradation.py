#!/usr/bin/env python3

import os
import cv2
import shutil
import numpy as np
from pathlib import Path

def apply_sensor_degradations(image):
    """
    Applies generalized optical and sensor degradations.
    Isolates sensor physics without introducing atmospheric variables.
    """
    # 1. Optical Lens Blur (Point Spread Function approximation)
    # A standard 3x3 Gaussian kernel removes artificial, crisp pixel edges
    img_blurred = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 2. Electronic Sensor Noise (Thermal/Shot Noise)
    # Generates a random static noise pattern across the channel array
    row, col, ch = img_blurred.shape
    mean = 0
    sigma = 12  # Noise standard deviation (adjust up for dirtier sensors)
    
    # Create the noise matrix matching the exact input dimensions
    gauss_noise = np.random.normal(mean, sigma, (row, col, ch)).astype(np.float32)
    
    # Add noise directly to the blurred array and clamp boundaries to valid 8-bit integers
    noisy_img = img_blurred.astype(np.float32) + gauss_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

def build_degraded_dataset(input_dir, output_dir):
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    
    if output_root.exists():
        print("🧹 Clearing previous degraded output tree...")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    valid_extensions = {".png", ".jpg", ".jpeg"}
    
    print(f"\n=== Running Isolated Sensor Degradation Pipeline ===")
    
    for split in splits:
        in_img_dir = input_root / "images" / split
        in_lbl_dir = input_root / "labels" / split
        out_img_dir = output_root / "images" / split
        out_lbl_dir = output_root / "labels" / split
        
        if not in_img_dir.exists():
            print(f"Skipping split [{split.upper()}] - Directory not found.")
            continue
            
        print(f"--> Degrading split: [{split.upper()}]")
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        image_list = sorted([f for f in os.listdir(in_img_dir) if Path(f).suffix.lower() in valid_extensions])
        total_files = len(image_list)
        
        for idx, img_name in enumerate(image_list):
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")
            
            # 1. Apply Sensor Physics Degradations to the Image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            degraded_img = apply_sensor_degradations(img)
            cv2.imwrite(str(out_img_dir / img_name), degraded_img)
            
            # 2. Map the Exact Labels (OBB Coordinates don't change, just the pixels)
            if lbl_path.exists():
                shutil.copy(lbl_path, out_lbl_dir / lbl_path.name)
                
            if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
                print(f"    Progress: [{idx + 1}/{total_files}] images degraded...", end="\r")
        print()

    print(f"✔ Dataset degradation complete!")
    
    # Generate the updated YOLO dataset config
    yaml_content = f"""path: {output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 1

# Classes
names:
  0: vehicle
"""
    yaml_path = output_root / "dataset.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml_file.write(yaml_content)
        
    print(f"📝 Generated updated configuration file at: {yaml_path}")
    print(f"🚀 Success! Sensor-degraded OBB dataset is ready at: {output_root}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # Setup your paths to mirror your workflow
    # ---------------------------------------------------------
    
    # Your clean dataset containing your 15GSD
    SRC_DATASET = "datasets_refined/cowc_512"
    
    # The target folder where the degraded images and pristine labels will be saved
    DEGRADED_DATASET = "datasets/cowc_512_sensor_degraded"
    
    build_degraded_dataset(SRC_DATASET, DEGRADED_DATASET)