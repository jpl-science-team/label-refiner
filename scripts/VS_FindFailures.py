#!/usr/bin/env python3

import os
import shutil
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

def obb_to_polygon(obb_coords, img_w=1024.0, img_h=1024.0):
    """Converts 8 normalized OBB coordinates [x1, y1, x2, y2, x3, y3, x4, y4] to a Shapely Polygon."""
    pts = np.array(obb_coords).reshape(4, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

def compute_obb_iou(poly1, poly2):
    """Computes the exact Intersection over Union between two oriented polygons."""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return inter_area / (union_area + 1e-6)
    except Exception:
        return 0.0

def load_yolo_obb_file(file_path):
    """Parses a YOLO OBB text file returning a list of dicts: {'class': int, 'poly': Polygon}."""
    targets = []
    if not file_path.exists():
        return targets
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:9]]
            poly = obb_to_polygon(coords)
            targets.append({'class': cls_id, 'poly': poly})
    return targets

def extract_failure_subset(gt_dir, pred_dir, output_subset_dir, iou_threshold=0.50):
    gt_root = Path(gt_dir)
    pred_root = Path(pred_dir)
    subset_root = Path(output_subset_dir)

    gt_img_dir = gt_root / "images"
    gt_lbl_dir = gt_root / "labels" if (gt_root / "labels").exists() else gt_root
    pred_lbl_dir = pred_root / "labels" if (pred_root / "labels").exists() else pred_root

    # Standard dataset directory structure
    out_img_dir = subset_root / "images"
    out_lbl_dir = subset_root / "labels"

    # Create destination directories
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    gt_files = list(gt_lbl_dir.glob("*.txt"))
    print(f"🔍 Scanning {len(gt_files)} image pairs for alignment failures (IoU < {iou_threshold})...")

    copied_images_count = 0
    total_failed_boxes_found = 0

    for gt_path in gt_files:
        base_name = gt_path.stem
        pred_path = pred_lbl_dir / gt_path.name
        
        # Look for the source image (.png, .jpg, or .jpeg)
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if (gt_img_dir / f"{base_name}{ext}").exists():
                img_path = gt_img_dir / f"{base_name}{ext}"
                break

        if not (pred_path.exists() and img_path and img_path.exists()):
            continue

        gt_boxes = load_yolo_obb_file(gt_path)
        pred_boxes = load_yolo_obb_file(pred_path)

        has_failure = False
        matched_preds = set()

        # Check if any ground truth box in this file fails the IoU threshold
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_pred_idx = -1

            for idx, pred_box in enumerate(pred_boxes):
                if idx in matched_preds:
                    continue
                iou = compute_obb_iou(gt_box['poly'], pred_box['poly'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = idx

            if best_iou >= iou_threshold:
                matched_preds.add(best_pred_idx)
            else:
                has_failure = True
                total_failed_boxes_found += 1

        # If this image contained at least 1 failing box, COPY the image and pipeline prediction
        if has_failure:
            copied_images_count += 1
            
            # COPY (not move) images and pipeline annotations (skipping ground truth)
            shutil.copy2(img_path, out_img_dir / img_path.name)
            shutil.copy2(pred_path, out_lbl_dir / pred_path.name)

    print(f"\n✅ Extraction Complete!")
    print(f"  - Copied {copied_images_count} image files with failures.")
    print(f"  - Total failed boxes isolated across these files: {total_failed_boxes_found}")
    print(f"  - Standardized subset ready at: {subset_root.resolve()}")

if __name__ == "__main__":
    # Your current paths
    HUMAN_GROUND_TRUTH = "datasets/vedai_1024/vedai_color_obb"
    PIPELINE_REFINED   = "datasets/vedai_1024/vedai_color_OBB_Refined"
    FAILURE_SUBSET_DIR = "datasets/vedai_1024/failures_only_subset"

    extract_failure_subset(
        gt_dir=HUMAN_GROUND_TRUTH,
        pred_dir=PIPELINE_REFINED,
        output_subset_dir=FAILURE_SUBSET_DIR,
        iou_threshold=0.50
    )