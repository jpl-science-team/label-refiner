#!/usr/bin/env python3
"""
Create a fixed size random subset of image/label pairs from each dataset.

This script:
- Searches each dataset for images located in images/train/ and images/val/
- Matches each image with its corresponding label in labels/train/ or labels/val/
- Randomly samples a specified number of complete image/label pairs
- Copies the sampled images into output/.../images/
- Copies the matching labels into output/.../labels/

"""


import os
import random
import shutil
from pathlib import Path

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def find_images(root):
    """Return list of (image_path, label_path) pairs inside train/ and val/."""
    pairs = []

    for split in ["train", "val"]:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            label_name = Path(img_name).stem + ".txt"
            label_path = lbl_dir / label_name

            if not label_path.exists():
                print(f"WARNING: missing label for {img_name} in {split}. Skipping.")
                continue

            pairs.append((img_dir / img_name, label_path))

    return pairs


def copy_subset(dataset_name, base_input, base_output, sample_count=200):
    print(f"\n=== DATASET: {dataset_name} ===")

    dataset_root = Path(base_input) / dataset_name
    if not dataset_root.exists():
        print(f"[SKIP] {dataset_root} does not exist.")
        return

    all_pairs = find_images(dataset_root)
    if len(all_pairs) == 0:
        print("[WARNING] No image/label pairs found.")
        return

    print(f"Found {len(all_pairs)} image/label pairs total.")

    n = min(sample_count, len(all_pairs))
    sampled = random.sample(all_pairs, n)

    out_img = Path(base_output) / dataset_name / "images"
    out_lbl = Path(base_output) / dataset_name / "labels"
    make_dir(out_img)
    make_dir(out_lbl)

    for img_path, lbl_path in sampled:
        shutil.copy2(img_path, out_img / img_path.name)
        shutil.copy2(lbl_path, out_lbl / lbl_path.name)

    print(f"Copied {n} samples for {dataset_name}.")


def main():
    random.seed(42)

    base_input = "datasets"
    base_output = "datasets/cowc_test_subset"

    datasets = ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]

    for d in datasets:
        copy_subset(d, base_input, base_output, sample_count=200)

    print("\nDone! Test subset created successfully.")
    print("Output: datasets/cowc_test_subset/")


if __name__ == "__main__":
    main()