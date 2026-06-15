#!/usr/bin/env python3

import os
import shutil
import random
from pathlib import Path
import yaml

# ---------------------------------------------------------
# Configuration (Mapped to your exact existing structure)
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where your current dataset is spread out
SRC_IMG_BASE = REPO_ROOT / "datasets_refined" / "cowc" / "images"
SRC_LBL_BASE = REPO_ROOT / "datasets_refined" / "cowc" / "labels"

# The pristine destination directory for your locked experiment
OUTPUT_BASE_DIR = REPO_ROOT / "processed_datasets" / "20260615_42_cowc_base"

# New locked target split ratios
SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.15,
    "test": 0.05
}

RANDOM_SEED = 42 

# ---------------------------------------------------------
# 1. Directory Setup
# ---------------------------------------------------------
def create_yolo_structure(base_dir):
    """Creates a clean, strict YOLOv8 directory structure."""
    base_path = Path(base_dir)
    if base_path.exists():
        print(f"[WARN] Output directory {base_dir} already exists. Overwriting...")
        shutil.rmtree(base_path)
    
    for split in ["train", "val", "test"]:
        (base_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    return base_path

# ---------------------------------------------------------
# 2. Main Processing Logic
# ---------------------------------------------------------
def lock_and_split_dataset():
    print("="*60)
    print(" 🔒 COWC DATASET UNIFICATION & LOCK INITIATED")
    print("="*60)
    
    all_valid_pairs = []
    
    # Scan through both existing subfolders to pool everything together
    current_subfolders = ["train", "val"]
    
    for folder in current_subfolders:
        img_dir = SRC_IMG_BASE / folder
        lbl_dir = SRC_LBL_BASE / folder
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] Skipping missing subdirectory: {folder}")
            continue
            
        print(f"Scanning input folder: {folder}...")
        for img_file in img_dir.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # Look for the matching label file in the paired labels folder
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    all_valid_pairs.append((img_file, lbl_file))

    total_images = len(all_valid_pairs)
    print(f"\nPooled a total of {total_images} valid image-label pairs from all directories.")
    
    if total_images == 0:
        print("[ERROR] No data pairs found. Please verify your source directories.")
        return

    # Shuffle everything cleanly using the locked seed to guarantee reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(all_valid_pairs)
    
    # Calculate target split indexes
    train_end = int(total_images * SPLIT_RATIOS["train"])
    val_end = train_end + int(total_images * SPLIT_RATIOS["val"])
    
    splits = {
        "train": all_valid_pairs[:train_end],
        "val": all_valid_pairs[train_end:val_end],
        "test": all_valid_pairs[val_end:]
    }
    
    output_path = create_yolo_structure(OUTPUT_BASE_DIR)
    
    # Copy files to their freshly locked split destinations
    print("\nDistributing files into new 80/15/5 folders...")
    for split_name, file_pairs in splits.items():
        print(f" -> Packaging {split_name.upper()}: {len(file_pairs)} files...")
        for img_src, lbl_src in file_pairs:
            img_dst = output_path / "images" / split_name / img_src.name
            lbl_dst = output_path / "labels" / split_name / lbl_src.name
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

    # ---------------------------------------------------------
    # 3. Generate clean data.yaml
    # ---------------------------------------------------------
    yaml_content = {
        "path": ".", 
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1, 
        "names": {
            0: "vehicle"
        }
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
    print("\n" + "="*50)
    print(" ✅ DATASET COMPLIANT & LOCKED")
    print("="*50)
    print(f" Exported location: {OUTPUT_BASE_DIR}")
    print(f" Final Train: {len(splits['train'])} images (80%)")
    print(f" Final Val:   {len(splits['val'])} images (15%)")
    print(f" Final Test:  {len(splits['test'])} images (5%)")
    print("="*50)
    print("You can now safely zip this processed folder up!")

if __name__ == "__main__":
    lock_and_split_dataset()