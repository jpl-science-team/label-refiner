#!/usr/bin/env python3

import os
from pathlib import Path

# ---------------------------------------------------------
# Configuration (Exact Relative Paths matching your script)
# ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# Source directories where the newly generated data lives
IMG_DIR = REPO_ROOT / "datasets_refined" / "cowc" / "images" / "val"
LBL_DIR = REPO_ROOT / "datasets_refined" / "cowc" / "labels" / "val"

# Blacklist directories containing the 712 files to exclude
REFINE_IMG_DIR = REPO_ROOT / "datasets_refined" / "cowc" / "needs_refinement" / "images"

def purge_blacklisted_files():
    print("="*60)
    print(" 🗑️  AUTOMATED DATASET PURGE INITIATED")
    print("="*60)

    # 1. Validation Checks
    if not REFINE_IMG_DIR.exists():
        print(f"[ERROR] Needs refinement folder not found at: {REFINE_IMG_DIR}")
        print("Please ensure your manual review files are placed there first.")
        return

    if not IMG_DIR.exists() or not LBL_DIR.exists():
        print(f"[ERROR] Target dataset directories missing.\nImg: {IMG_DIR}\nLbl: {LBL_DIR}")
        return

    # 2. Gather the filenames of the 712 bad images
    blacklisted_filenames = [
        f.name for f in REFINE_IMG_DIR.glob("*.*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    total_blacklist = len(blacklisted_filenames)
    print(f"Identified {total_blacklist} files in 'needs_refinement' to purge.")

    if total_blacklist == 0:
        print("No files found in 'needs_refinement'. Nothing to do!")
        return

    # 3. Process Deletions
    purged_images = 0
    purged_labels = 0
    skipped_count = 0

    print("\nScanning and removing files from train directories...")
    for img_name in blacklisted_filenames:
        target_img_file = IMG_DIR / img_name
        target_lbl_file = LBL_DIR / (Path(img_name).stem + ".txt")
        
        file_found = False

        # Remove image if it exists in train
        if target_img_file.exists():
            target_img_file.unlink()
            purged_images += 1
            file_found = True
            
        # Remove matching label if it exists in train
        if target_lbl_file.exists():
            target_lbl_file.unlink()
            purged_labels += 1
            file_found = True

        if not file_found:
            skipped_count += 1

    # 4. Final Output Report
    print("-" * 60)
    print("✅ Purge Operation Complete!")
    print(f" -> Removed from images/train: {purged_images} files")
    print(f" -> Removed from labels/train: {purged_labels} files")
    if skipped_count > 0:
        print(f" -> Files already absent from train sets: {skipped_count}")
    print("-" * 60)
    
    remaining_clean_images = len(list(IMG_DIR.glob("*.*")))
    print(f"Remaining 'gold standard' baseline images: {remaining_clean_images}")
    print("="*60)

if __name__ == "__main__":
    purge_blacklisted_files()