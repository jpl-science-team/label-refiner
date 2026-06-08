#!/usr/bin/env python3

"""
Refine YOLO bounding boxes using the Segment Anything Model (SAM).

This script takes an existing dataset of images and YOLO-format labels,
uses SAM to refine each bounding box into a tighter, mask derived box,
and writes out an improved version of the dataset.

----------------------------------------------------------------------
What the script actually does
----------------------------------------------------------------------

1. Loads a SAM model checkpoint and initializes a SamPredictor on the
   best available device (MPS → CUDA → CPU).

2. Iterates over each dataset in:
       ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]

3. For each dataset, it expects this input directory structure:
       <input_root>/<dataset>/images/*.png
       <input_root>/<dataset>/labels/*.txt
   (No train/val splits — this script operates on a single flat folder.)

4. For each image:
   - Loads the image from the images/ directory.
   - Copies the image *unchanged* to the output directory.
   - Loads the corresponding YOLO label file.
   - Sends the image to SAM once per image.

5. For each YOLO bounding box:
   - Converts normalized center format YOLO coordinates to absolute pixels.
   - Creates a SAM prompt using:
         - A point at the YOLO box center
         - The YOLO box as a bounding box prompt
   - Runs SAM to obtain a segmentation mask.
   - Converts the mask back into an axis aligned YOLO bounding box.
   - If SAM fails, the original YOLO box is kept.

6. Writes a new .txt label file for each image with the refined boxes.

7. Outputs the refined datasets to:
       <output_root>/<dataset>/images/
       <output_root>/<dataset>/labels/

----------------------------------------------------------------------
Important notes
----------------------------------------------------------------------

- This script refines bounding boxes only — it does NOT produce masks.
- It assumes every YOLO box represents a single object.
- If SAM returns no segmentation for a box, the original label is preserved.
- The script operates on a flat images/ and labels/ structure (no splits).
- Output labels are written with high precision for downstream training.

----------------------------------------------------------------------
Usage
----------------------------------------------------------------------

The main entry point loads a SAM checkpoint and refines all test-subset
datasets located under:

    input_root="datasets/cowc_test_subset"

Results are written to:

    output_root="datasets/cowc_test_refined"

"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# Load SAM model
# ---------------------------------------------------------
def load_sam(model_path="models/sam_hq_vit_b.pth"):
    print(f"Loading SAM checkpoint from: {model_path}")

    sam = sam_model_registry["vit_b"](checkpoint=model_path)

    # Use MPS for Mac, CUDA for Nvidia, otherwise CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    sam.to(device)

    predictor = SamPredictor(sam)
    print(f"SAM loaded on {device}")
    return predictor, device


# ---------------------------------------------------------
# Convert mask → standard axis-aligned YOLO box
# ---------------------------------------------------------
def mask_to_yolo_box(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask.astype("uint8"),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    # x, y is top-left corner, w, h is width and height of axis-aligned box
    x, y, bw, bh = cv2.boundingRect(cnt)

    # Convert to YOLO format (normalized center x, center y, width, height)
    cx = (x + bw / 2.0) / img_w
    cy = (y + bh / 2.0) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h

    return cx, cy, w_norm, h_norm


# ---------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------
def refine_dataset(input_root, output_root, predictor):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]

    for dataset in datasets:
        print(f"\n=== Refining dataset: {dataset} ===")

        in_img_dir = input_root / dataset / "images"
        in_lbl_dir = input_root / dataset / "labels"

        out_img_dir = output_root / dataset / "images"
        out_lbl_dir = output_root / dataset / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not in_img_dir.exists():
            print(f"[WARN] Directory {in_img_dir} does not exist, skipping dataset.")
            continue

        image_list = sorted([f for f in os.listdir(in_img_dir)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        for img_name in image_list:
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                print(f"[WARN] Missing label for {img_name}, skipping.")
                continue

            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Failed to read image {img_name}, skipping.")
                continue
            img_h, img_w = img.shape[:2]

            # Copy image unchanged
            cv2.imwrite(str(out_img_dir / img_name), img)

            # Load YOLO labels
            with open(lbl_path) as f:
                lines = f.read().strip().splitlines()

            refined_labels = []

            # Set image in SAM predictor once per image
            predictor.set_image(img)

            for line in lines:
                parts = list(map(float, line.split()))
                if len(parts) < 5:
                    continue
                cls_id, cx, cy, w, h = parts[:5]

                # Convert normalized coordinates to absolute pixel values (with rounding)
                px = int(round(cx * img_w))
                py = int(round(cy * img_h))
                
                # Derive absolute box coordinates to use as a prompt for SAM
                x1 = int(round((cx - w / 2.0) * img_w))
                y1 = int(round((cy - h / 2.0) * img_h))
                x2 = int(round((cx + w / 2.0) * img_w))
                y2 = int(round((cy + h / 2.0) * img_h))

                input_point = np.array([[px, py]])
                input_label = np.array([1])
                input_box = np.array([x1, y1, x2, y2])

                # Passing both the box and point prompt forces SAM to be incredibly accurate
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box[None, :],  # Shape must be [1, 4]
                    multimask_output=False
                )

                mask = masks[0]
                yolo_box = mask_to_yolo_box(mask, img_w, img_h)

                if yolo_box is None:
                    # fallback: Keep original label if segmentation fails completely
                    refined_labels.append([int(cls_id), cx, cy, w, h])
                    continue

                cx_n, cy_n, bw_n, bh_n = yolo_box
                refined_labels.append([int(cls_id), cx_n, cy_n, bw_n, bh_n])

            # Save refined label file
            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for r in refined_labels:
                    # Write class as integer, coordinates with high precision
                    f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")

        print(f"✔ Finished refining dataset: {dataset}")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    predictor, device = load_sam("models/sam_vit_b.pth")

    refine_dataset(
        input_root="datasets/cowc_test_subset",
        output_root="datasets/cowc_test_refined",
        predictor=predictor
    )