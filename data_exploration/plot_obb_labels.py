#!/usr/bin/env python3
"""
Batch renderer that draws YOLO OBB labels over their images and saves PNGs.
If refinement_report.csv (written by refine_labels.py) sits next to the
labels folder, boxes are colored by how they were produced:
- Green   : SAM mask accepted
- Cyan    : replaced by the geometric correction head (size/angle borrowed from a nearby car)
- Orange  : fallback, SAM mask rejected and the original placeholder box kept
- Magenta X: low similarity to the reference (flagged for review)
- Blue dot : SAM prompt point, when snapping moved it away from the label point
- Red dot  : original point label center (if --input-labels is given)
Without a report, rotated boxes are green and axis-aligned boxes orange.

Example:
    python data_exploration/plot_obb_labels.py \
        --images COWC_mini_test/mini_dataset_refined/images --labels COWC_mini_test/mini_dataset_refined/labels \
        --input-labels COWC_mini_test/mini_dataset/labels --out COWC_mini_test/mini_dataset_refined/plots
"""

import argparse
import csv
import cv2
import numpy as np
from pathlib import Path

# Configurable Display Options
SCALE = 2  # upscale factor so ~30px cars are easy to see
COLOR_ROTATED = (0, 255, 0)      # Green
COLOR_AXIS = (0, 165, 255)       # Orange
COLOR_CORRECTED = (255, 255, 0)  # Cyan
COLOR_POINT = (0, 0, 255)        # Red
COLOR_PROMPT = (255, 0, 0)       # Blue
COLOR_LOW_SIM = (255, 0, 255)    # Magenta
SOURCE_COLORS = {"sam": COLOR_ROTATED, "corrected": COLOR_CORRECTED, "fallback": COLOR_AXIS}


def read_labels(path):
    if not path.exists():
        return []
    return [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]


def is_axis_aligned(pts):
    xs, ys = np.round(pts[:, 0], 3), np.round(pts[:, 1], 3)
    return len(set(xs)) <= 2 and len(set(ys)) <= 2


def read_report(path):
    """Groups refinement_report.csv rows by image, keeping only boxes written to the label files."""
    report = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("kept", "1") == "1":
                    report.setdefault(Path(row["image"]).stem, []).append(row)
    return report


def draw_image(img, obb_labels, point_labels, box_info=None):
    h, w = img.shape[:2]
    canvas = cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
    if box_info is not None and len(box_info) != len(obb_labels):
        box_info = None  # report doesn't match these labels; fall back to shape-based colors

    for i, lbl in enumerate(obb_labels):
        if len(lbl) < 9:
            continue
        pts = np.array(lbl[1:9]).reshape(4, 2) * np.array([w, h])
        if box_info:
            color = SOURCE_COLORS.get(box_info[i]["source"], COLOR_ROTATED)
        else:
            color = COLOR_AXIS if is_axis_aligned(pts) else COLOR_ROTATED
        cv2.polylines(canvas, [np.round(pts * SCALE).astype(np.int32)], True, color, 1, cv2.LINE_AA)

        if box_info:
            info = box_info[i]
            lx, ly = int(info["label_x"]) * SCALE, int(info["label_y"]) * SCALE
            qx, qy = int(info["prompt_x"]) * SCALE, int(info["prompt_y"]) * SCALE
            if (qx, qy) != (lx, ly):
                cv2.line(canvas, (lx, ly), (qx, qy), COLOR_PROMPT, 1, cv2.LINE_AA)
                cv2.circle(canvas, (qx, qy), 2, COLOR_PROMPT, -1)
            if info["low_similarity"] == "1":
                c = np.round(pts.mean(axis=0) * SCALE).astype(int)
                cv2.drawMarker(canvas, (int(c[0]), int(c[1])), COLOR_LOW_SIM, cv2.MARKER_TILTED_CROSS, 14, 2)

    for lbl in point_labels:
        cx, cy = lbl[1] * w * SCALE, lbl[2] * h * SCALE
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), 2, COLOR_POINT, -1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="YOLO OBB label dir")
    parser.add_argument("--input-labels", type=Path, help="Original point label dir (optional)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="refinement_report.csv (default: next to the labels folder)")
    args = parser.parse_args()

    report = read_report(args.report or args.labels.parent / "refinement_report.csv")

    args.out.mkdir(parents=True, exist_ok=True)
    image_list = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

    for img_path in image_list:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARNING] Skipping {img_path.name}: Unable to read image file.")
            continue
        obb = read_labels(args.labels / (img_path.stem + ".txt"))
        points = read_labels(args.input_labels / (img_path.stem + ".txt")) if args.input_labels else []
        box_info = report.get(img_path.stem) if report else None
        cv2.imwrite(str(args.out / (img_path.stem + ".png")), draw_image(img, obb, points, box_info))

    print(f"✔ Wrote {len(image_list)} plots to {args.out}")


if __name__ == "__main__":
    main()
