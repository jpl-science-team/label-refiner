#!/usr/bin/env python3

"""
Run GroundingDINO object detection over the COWC test-subset datasets.

This script performs the following steps:

1. Loads GroundingDINO model + config.
2. Processes the same datasets as your SAM script:
       ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]
3. For each image:
       - Loads YOLO labels (to get center points)
       - Crops a small tile around each center
       - Runs GroundingDINO to detect vehicles inside the crop
       - Converts DETECTIONS back to full-image YOLO coords
4. Writes refined YOLO-format detection labels.

This script **does not** run SAM — this version is ONLY for validating
GroundingDINO detections before merging with SAM.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path

from groundingdino.util.inference import Model
from groundingdino.util.inference import load_model, predict, annotate


# ---------------------------------------------------------------------------
# Load GroundingDINO model
# ---------------------------------------------------------------------------
def load_grounding_dino(
    config_path="external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    weights_path="models/groundingdino/groundingdino_swint_ogc.pth"
):
    print("\nLoading GroundingDINO...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=weights_path,
        device=device
    )
    print(f"GroundingDINO loaded on: {device}")
    return model, device


# ---------------------------------------------------------------------------
# Run GroundingDINO on a cropped tile
# ---------------------------------------------------------------------------
def run_dino_on_crop(model, crop_img, text_prompt="vehicle"):
    """
    Runs GroundingDINO detection on a cropped tile.

    Returns list of YOLO-format detections in crop-relative coords.
    """
    detections = model.predict_with_classes(
        image=crop_img,
        classes=[text_prompt],
        box_threshold=0.25,
        text_threshold=0.25
    )

    yolo_boxes = []

    h, w = crop_img.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["box"]

        # YOLO normalized crop coords
        cx = (x1 + (x2 - x1) / 2) / w
        cy = (y1 + (y2 - y1) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        yolo_boxes.append((0, cx, cy, bw, bh))  # class 0 hard-coded for now

    return yolo_boxes


# ---------------------------------------------------------------------------
# Convert crop-relative YOLO boxes → global YOLO boxes
# ---------------------------------------------------------------------------
def crop_to_full_image(box, crop_x, crop_y, crop_w, crop_h, img_w, img_h):
    cls_id, cx, cy, bw, bh = box

    # convert crop-relative back to absolute pixels
    abs_cx = crop_x + cx * crop_w
    abs_cy = crop_y + cy * crop_h
    abs_w = bw * crop_w
    abs_h = bh * crop_h

    # normalize to full-image YOLO coords
    return (
        cls_id,
        abs_cx / img_w,
        abs_cy / img_h,
        abs_w / img_w,
        abs_h / img_h
    )


# ---------------------------------------------------------------------------
# Main dataset loop
# ---------------------------------------------------------------------------
def run_grounding_dino(input_root, output_root, model):
    datasets = ["cowc", "cowc_rgb_1m", "cowc_rgb_05m"]

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        print(f"\n=== Running GroundingDINO on dataset: {dataset} ===")

        in_img_dir = input_root / dataset / "images"
        in_lbl_dir = input_root / dataset / "labels"

        out_img_dir = output_root / dataset / "images"
        out_lbl_dir = output_root / dataset / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not in_img_dir.exists():
            print(f"[WARN] Missing {in_img_dir}, skipping.")
            continue

        image_list = sorted([f for f in os.listdir(in_img_dir)
                             if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        for img_name in image_list:
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                print(f"[WARN] Missing label file for {img_name}, skipping.")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Could not load image {img_path}")
                continue

            img_h, img_w = img.shape[:2]
            cv2.imwrite(str(out_img_dir / img_name), img)  # copy image

            # load center points
            with open(lbl_path, "r") as f:
                labels = f.read().strip().splitlines()

            all_global_boxes = []

            for line in labels:
                parts = list(map(float, line.split()))
                if len(parts) < 5:
                    continue

                _, cx, cy, w, h = parts

                px = int(cx * img_w)
                py = int(cy * img_h)

                crop_size = 256
                x1 = max(0, px - crop_size // 2)
                y1 = max(0, py - crop_size // 2)
                x2 = min(img_w, x1 + crop_size)
                y2 = min(img_h, y1 + crop_size)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # run GroundingDINO
                crop_boxes = run_dino_on_crop(model, crop)

                # convert detections → full image coords
                for b in crop_boxes:
                    full_box = crop_to_full_image(
                        b,
                        crop_x=x1,
                        crop_y=y1,
                        crop_w=(x2 - x1),
                        crop_h=(y2 - y1),
                        img_w=img_w,
                        img_h=img_h
                    )
                    all_global_boxes.append(full_box)

            # write output YOLO labels
            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for cls_id, cx, cy, bw, bh in all_global_boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"✔ Finished dataset {dataset}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, device = load_grounding_dino()

    run_grounding_dino(
        input_root="datasets/cowc_test_subset",
        output_root="datasets/cowc_test_gdino",
        model=model,
    )