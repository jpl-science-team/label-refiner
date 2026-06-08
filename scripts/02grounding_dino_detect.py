#!/usr/bin/env python3

"""
Optimized GroundingDINO vehicle detection for Tesla M60 GPUs.

Features:
    • FP16 inference (with safe fallback to FP32)
    • Multi-GPU support (CUDA_VISIBLE_DEVICES from SLURM script)
    • Aggressive tiling (crop=256, change at top if needed)
    • No flags, no CLI args — clean and self-contained
    • Writes only YOLO boxes, no extra output
    • Lightweight memory footprint for 8GB GPUs
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path

from groundingdino.util.inference import Model

# -----------------------------------------------------------
# User-edited VARIABLES (no flags, no CLI args)
# -----------------------------------------------------------
CROP_SIZE = 256          # Aggressive tiling; change manually if needed
TEXT_PROMPT = "vehicle"  # GroundingDINO class name
OUTPUT_THRESHOLD = 0.25  # Box/text confidence
USE_FP16 = True          # Force FP16 inference on M60
# -----------------------------------------------------------


# -----------------------------------------------------------
# Load GroundingDINO Model (FP16-safe)
# -----------------------------------------------------------
def load_grounding_dino(
    config_path="external/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    weights_path="models/groundingdino/groundingdino_swint_ogc.pth",
):
    print("\n[INFO] Loading GroundingDINO...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(
        model_config_path=config_path,
        model_checkpoint_path=weights_path,
        device=device,
    )

    # -------------------------------
    # FP16 Mode with safe fallback
    # -------------------------------
    if USE_FP16 and device == "cuda":
        try:
            model.model.half()
            print("[INFO] Running model in FP16 mode")
        except Exception as e:
            print("[WARN] FP16 conversion failed, falling back to FP32:", e)

    print(f"[INFO] GroundingDINO loaded on: {device}")
    return model, device


# -----------------------------------------------------------
# Run GroundingDINO on a crop (FP16 friendly)
# -----------------------------------------------------------
@torch.inference_mode()
def run_dino_on_crop(model, crop_img):
    """
    Returns list of YOLO-format detections (class, cx, cy, w, h).
    """

    # FP16 autocast only on CUDA
    with torch.cuda.amp.autocast(enabled=(USE_FP16 and torch.cuda.is_available())):
        detections = model.predict_with_classes(
            image=crop_img,
            classes=[TEXT_PROMPT],
            box_threshold=OUTPUT_THRESHOLD,
            text_threshold=OUTPUT_THRESHOLD,
        )

    yolo_boxes = []
    h, w = crop_img.shape[:2]

    for d in detections:
        x1, y1, x2, y2 = d["box"]

        cx = (x1 + (x2 - x1) / 2) / w
        cy = (y1 + (y2 - y1) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        yolo_boxes.append((0, cx, cy, bw, bh))  # class 0 fixed

    return yolo_boxes


# -----------------------------------------------------------
# Convert from crop coordinates to image coordinates
# -----------------------------------------------------------
def crop_to_full_image(box, crop_x, crop_y, crop_w, crop_h, img_w, img_h):
    cls_id, cx, cy, bw, bh = box

    abs_cx = crop_x + cx * crop_w
    abs_cy = crop_y + cy * crop_h
    abs_w = bw * crop_w
    abs_h = bh * crop_h

    return (
        cls_id,
        abs_cx / img_w,
        abs_cy / img_h,
        abs_w / img_w,
        abs_h / img_h,
    )


# -----------------------------------------------------------
# Main dataset loop (unchanged logic, optimized internals)
# -----------------------------------------------------------
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

        image_list = sorted(
            f for f in os.listdir(in_img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        for img_name in image_list:
            img_path = in_img_dir / img_name
            lbl_path = in_lbl_dir / (Path(img_name).stem + ".txt")

            if not lbl_path.exists():
                print(f"[WARN] Missing label for {img_name}, skipping.")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            cv2.imwrite(str(out_img_dir / img_name), img)

            # Load YOLO center points
            with open(lbl_path, "r") as f:
                labels = f.read().strip().splitlines()

            all_detections = []

            for line in labels:
                parts = list(map(float, line.split()))
                if len(parts) < 5:
                    continue

                _, cx, cy, w, h = parts

                px = int(cx * img_w)
                py = int(cy * img_h)

                x1 = max(0, px - CROP_SIZE // 2)
                y1 = max(0, py - CROP_SIZE // 2)
                x2 = min(img_w, x1 + CROP_SIZE)
                y2 = min(img_h, y1 + CROP_SIZE)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_boxes = run_dino_on_crop(model, crop)

                for b in crop_boxes:
                    full_box = crop_to_full_image(
                        b, x1, y1, (x2 - x1), (y2 - y1), img_w, img_h
                    )
                    all_detections.append(full_box)

            # Write YOLO output
            out_lbl_path = out_lbl_dir / (Path(img_name).stem + ".txt")
            with open(out_lbl_path, "w") as f:
                for cls_id, cx, cy, bw, bh in all_detections:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"✔ Finished dataset {dataset}")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    model, device = load_grounding_dino()

    run_grounding_dino(
        input_root="datasets/cowc_test_subset",
        output_root="datasets/cowc_test_gdino",
        model=model,
    )