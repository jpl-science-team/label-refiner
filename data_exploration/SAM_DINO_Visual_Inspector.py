#!/usr/bin/env python3
"""
Interactive Dual-Label Viewer for comparing VEDAI (GT) vs SAM DINO (Pipeline) annotations.
- Blue/Cyan  : VEDAI Ground Truth Labels
- Magenta/Pink: SAM DINO Pipeline Labels
- Text Labels: Oriented IoU match percentage
- Controls   : Right/Left Arrow (or A/D) to navigate, Q or ESC to quit.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

# Configurable Display Options
WINDOW_NAME = "VEDAI vs SAM DINO Visual Inspector"
ZOOM_SIZE = 80
ZOOM_SCALE = 8
DISPLAY_HEIGHT = 800

# Color Codes (BGR Format)
COLOR_VEDAI = (255, 255, 0)     # Cyan / Light Blue (Human GT)
COLOR_SAM_DINO = (255, 0, 255)  # Magenta / Bright Pink (Pipeline)

mouse_pos = [0, 0]
scale_factor = 1.0


def mouse_callback(event, x, y, flags, param):
    global mouse_pos, scale_factor
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos[0] = int(x / scale_factor)
        mouse_pos[1] = int(y / scale_factor)


def compute_obb_iou(poly1, poly2):
    """Computes the exact Intersection over Union between two polygons."""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        return inter_area / (union_area + 1e-6)
    except Exception:
        return 0.0


def load_boxes(lbl_path, w, h):
    """Loads YOLO labels and converts them into Shapely Polygons and OpenCV points."""
    boxes = []
    if not lbl_path.exists():
        return boxes

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            vals = list(map(float, parts[1:]))

            # Oriented Bounding Boxes (8 coordinates)
            if len(vals) == 8:
                pts = np.array(
                    [[vals[i] * w, vals[i + 1] * h] for i in range(0, 8, 2)],
                    np.int32,
                )
            # Standard Bounding Boxes (4 coordinates)
            elif len(vals) == 4:
                cx, cy, bw, bh = vals
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
            else:
                continue

            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            boxes.append({'pts': pts, 'poly': poly})
            
    return boxes


def draw_matched_boxes(img, gt_boxes, pred_boxes):
    """Matches GT and Pred boxes, drawing them with IoU labels."""
    
    # 1. Draw all Ground Truth boxes first (so they sit underneath)
    for gt in gt_boxes:
        cv2.polylines(img, [gt['pts'].reshape((-1, 1, 2))], True, COLOR_VEDAI, 2)

    # 2. Match predictions to Ground Truths
    matched_preds = set()
    for gt in gt_boxes:
        best_iou = 0.0
        best_pred_idx = -1
        
        for idx, pred in enumerate(pred_boxes):
            if idx in matched_preds:
                continue
            iou = compute_obb_iou(gt['poly'], pred['poly'])
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = idx

        if best_iou > 0 and best_pred_idx != -1:
            matched_preds.add(best_pred_idx)
            pred_boxes[best_pred_idx]['iou'] = best_iou

    # 3. Draw Predictions and Labels
    for idx, pred in enumerate(pred_boxes):
        iou = pred.get('iou', 0.0)
        pts = pred['pts']
        
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], True, COLOR_SAM_DINO, 2)
        
        # Find the highest point of the polygon to anchor the text cleanly
        top_point_idx = np.argmin(pts[:, 1])
        tx = int(pts[top_point_idx][0])
        ty = max(15, int(pts[top_point_idx][1]) - 5)
        
        text = f"{iou:.2f}"
        
        # Draw text with a black outline for contrast against light/dark backgrounds
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def main():
    global scale_factor

    # Set up your directory paths (matching your environment)
    VEDAI_ROOT = Path("datasets/vedai_1024/vedai_color_obb")
    SAM_DINO_ROOT = Path("datasets/vedai_1024/vedai_color_OBB_Refined")

    img_dir = VEDAI_ROOT / "images"
    vedai_lbl_dir = VEDAI_ROOT / "labels"
    sam_dino_lbl_dir = SAM_DINO_ROOT / "labels"

    # Fallback if labels directory layout differs slightly
    if not vedai_lbl_dir.exists():
        vedai_lbl_dir = VEDAI_ROOT
    if not sam_dino_lbl_dir.exists():
        sam_dino_lbl_dir = SAM_DINO_ROOT

    if not img_dir.exists():
        print(f"❌ Error: Image directory not found at {img_dir.resolve()}")
        return

    img_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    if not img_files:
        print("❌ Error: No images found in the target directory.")
        return

    idx = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("🚀 Inspector Started!")
    print("Controls:")
    print("  ➡ / D : Next Image")
    print("  ⬅ / A : Previous Image")
    print("  Q / ESC : Exit")
    print("\nLegend:")
    print("  Cyan Line    : VEDAI Ground Truth")
    print("  Magenta Line : SAM DINO Pipeline Output")
    print("  White Text   : Oriented Intersection over Union (IoU)")

    while True:
        file_name = img_files[idx]
        base_stem = Path(file_name).stem

        img_path = img_dir / file_name
        vedai_lbl_path = vedai_lbl_dir / f"{base_stem}.txt"
        sam_dino_lbl_path = sam_dino_lbl_dir / f"{base_stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            idx = (idx + 1) % len(img_files)
            continue

        h, w = img.shape[:2]
        canvas = img.copy()

        # Load geometries and draw them mapped with IoU text
        gt_boxes = load_boxes(vedai_lbl_path, w, h)
        pred_boxes = load_boxes(sam_dino_lbl_path, w, h)
        canvas = draw_matched_boxes(canvas, gt_boxes, pred_boxes)

        # Standardize scaling relative to display height
        scale_factor = DISPLAY_HEIGHT / h

        # Calculate mouse cursor coordinates inside original resolution
        mx, my = mouse_pos
        mx = int(np.clip(mx, 0, w - 1))
        my = int(np.clip(my, 0, h - 1))

        # Crop Region of Interest for the Live Zoom Window
        x1, y1 = max(mx - ZOOM_SIZE // 2, 0), max(my - ZOOM_SIZE // 2, 0)
        x2, y2 = min(mx + ZOOM_SIZE // 2, w), min(my + ZOOM_SIZE // 2, h)
        roi = canvas[y1:y2, x1:x2]

        # Rescale zoomed window
        zoom = cv2.resize(roi, (h, h), interpolation=cv2.INTER_NEAREST)

        # Draw visual HUD headers
        cv2.putText(canvas, f"[{idx+1}/{len(img_files)}] {file_name}", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(zoom, "CYAN: VEDAI | MAGENTA: SAM DINO", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Merge split viewer (Annotated Image + Zoom Panel)
        combined = np.hstack([canvas, zoom])
        disp_w = int(combined.shape[1] * scale_factor)
        frame = cv2.resize(combined, (disp_w, DISPLAY_HEIGHT))

        cv2.imshow(WINDOW_NAME, frame)

        # Capture keypresses across platforms
        key = cv2.waitKeyEx(30)

        # Exit Keys (Q, ESC)
        if key in (ord("q"), ord("Q"), 27):
            break

        # Next Image Keys (Right Arrow Key codes or 'd')
        elif key in (ord("d"), ord("D"), 2555904, 83, 65363, 3):
            idx = (idx + 1) % len(img_files)

        # Previous Image Keys (Left Arrow Key codes or 'a')
        elif key in (ord("a"), ord("A"), 2424832, 81, 65361, 2):
            idx = (idx - 1) % len(img_files)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()