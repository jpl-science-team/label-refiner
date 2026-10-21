#!/usr/bin/env python3
"""
Scores refined OBB labels (from refine_labels.py) against the true VEDAI OBBs
(car, van and pickup, as kept by VS_VEDAI_Separation_Conversion.py).

Every refined box comes from a true vehicle's center point, so the question is box quality:
each true vehicle is matched one-to-one to a refined box by oriented IoU (greedy, highest IoU
first) and the script reports IoU, size and angle errors, overall and per `source`
(sam / corrected / fallback, read from refinement_report.csv when present).

Holdout: images in VEDAI's official foldNNtest.txt list are the holdout set; every other
image is the tuning set. Tune parameters with --split tune and check the final settings
once with --split holdout.

Example:
  python scripts/refine_labels.py --input datasets/vedai_512/vedai_color_centerpoint \
      --output datasets_refined/vedai_color --box-prompt
  python scripts/evaluate_updated_vedai_labels.py --pred datasets_refined/vedai_color --split tune
"""

import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import Polygon

IMG_SIZE = 512.0             # VEDAI 512 tiles; labels are normalized to this
IOU_THRESHOLDS = (0.5, 0.7)

def load_obbs(lbl_path):
    """Reads YOLO OBB lines (class x1 y1 ... x4 y4, normalized) as (4, 2) pixel arrays."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9:
            boxes.append(np.array(parts[1:9], dtype=float).reshape(4, 2) * IMG_SIZE)
    return boxes

def to_polygon(pts):
    poly = Polygon(pts)
    return poly if poly.is_valid else poly.buffer(0)

def obb_iou(p1, p2):
    inter = p1.intersection(p2).area
    return inter / (p1.area + p2.area - inter + 1e-6)

def rect_params(pts):
    """(length, width, angle in [0, 180)) of the box's long axis."""
    (_, _), (w, h), angle = cv2.minAreaRect(pts.astype(np.float32))
    if w < h:
        w, h, angle = h, w, angle + 90.0
    return w, h, angle % 180.0

def angle_error(a1, a2):
    d = abs(a1 - a2) % 180.0
    return min(d, 180.0 - d)

def load_sources(pred_dir):
    """Maps image name -> list of `source` for each box written to its label file, in order."""
    report = pred_dir / "refinement_report.csv"
    sources = {}
    if not report.exists():
        return sources
    with open(report) as f:
        for row in csv.DictReader(f):
            if row["kept"] == "1":
                sources.setdefault(Path(row["image"]).stem, []).append(row["source"])
    return sources

def match(truth_polys, pred_polys):
    """Greedy one-to-one matching by IoU. Returns [(truth_idx, pred_idx, iou)] with iou > 0."""
    pairs = [(obb_iou(t, p), i, j) for i, t in enumerate(truth_polys) for j, p in enumerate(pred_polys)]
    pairs.sort(reverse=True)
    used_t, used_p, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou <= 0 or i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matches.append((i, j, iou))
    return matches

def summarize(rows, n_truth, label):
    """One block of summary text for a group of per-car rows."""
    ious = np.array([r["iou"] for r in rows]) if rows else np.zeros(0)
    lines = [f"{label}: {n_truth} true vehicles"]
    if n_truth == 0:
        return lines
    lines.append(f"  mean IoU {ious.sum() / n_truth:.3f} (unmatched count as 0), "
                 f"median IoU {np.median(np.concatenate([ious, np.zeros(n_truth - len(ious))])):.3f}")
    lines.append("  " + ", ".join(f"IoU>={t}: {100 * (ious >= t).sum() / n_truth:.1f}%" for t in IOU_THRESHOLDS))
    matched = [r for r in rows if r["iou"] > 0]
    if matched:
        lines.append(f"  matched boxes: median length error {np.median([r['d_length'] for r in matched]):+.1f}px, "
                     f"width error {np.median([r['d_width'] for r in matched]):+.1f}px, "
                     f"angle error {np.median([r['angle_err'] for r in matched]):.1f} deg")
    return lines

def main():
    parser = argparse.ArgumentParser(description="Score refined VEDAI OBB labels against the true vehicle OBBs.")
    parser.add_argument("--pred", required=True, help="refine_labels.py output dir (with labels/ and refinement_report.csv)")
    parser.add_argument("--truth", default="datasets/vedai_512/vedai_color_obb/labels",
                        help="True OBB labels written by VS_VEDAI_Separation_Conversion.py")
    parser.add_argument("--folds", default="data/VEDAI/Annotations512", help="Dir with VEDAI's foldNNtest.txt lists")
    parser.add_argument("--fold", type=int, default=1, help="Which foldNNtest.txt is the holdout set (default 1)")
    parser.add_argument("--split", choices=["tune", "holdout", "all"], default="tune",
                        help="tune = all images outside the holdout fold (default), holdout = the fold's test images")
    parser.add_argument("--out", default=None, help="Where to write the per-car CSV and summary (default: --pred)")
    args = parser.parse_args()

    pred_dir, truth_dir = Path(args.pred), Path(args.truth)
    pred_lbl_dir = pred_dir / "labels" if (pred_dir / "labels").is_dir() else pred_dir
    out_dir = Path(args.out) if args.out else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout = set((Path(args.folds) / f"fold{args.fold:02d}test.txt").read_text().split())
    truth_files = sorted(truth_dir.glob("*.txt"))
    # Truth files are named like 00000010_co.txt; the fold lists use the bare id
    in_holdout = lambda p: p.stem.split("_")[0] in holdout
    if args.split == "tune":
        truth_files = [p for p in truth_files if not in_holdout(p)]
    elif args.split == "holdout":
        truth_files = [p for p in truth_files if in_holdout(p)]

    sources = load_sources(pred_dir)
    rows, n_truth, n_pred, n_extra, missing_files = [], 0, 0, 0, 0

    for truth_path in truth_files:
        truth = load_obbs(truth_path)
        pred_path = pred_lbl_dir / truth_path.name
        if not pred_path.exists():
            missing_files += 1
        preds = load_obbs(pred_path)
        n_truth += len(truth)
        n_pred += len(preds)
        img_sources = sources.get(truth_path.stem, [])

        matches = match([to_polygon(t) for t in truth], [to_polygon(p) for p in preds])
        n_extra += len(preds) - len(matches)
        matched = {i: (j, iou) for i, j, iou in matches}

        for i, t in enumerate(truth):
            t_len, t_wid, t_ang = rect_params(t)
            row = {"image": truth_path.stem, "car": i, "true_length": round(t_len, 1), "true_width": round(t_wid, 1)}
            if i in matched:
                j, iou = matched[i]
                p_len, p_wid, p_ang = rect_params(preds[j])
                row.update(iou=iou, source=img_sources[j] if j < len(img_sources) else "unknown",
                           d_length=p_len - t_len, d_width=p_wid - t_wid, angle_err=angle_error(p_ang, t_ang))
            else:
                row.update(iou=0.0, source="unmatched", d_length="", d_width="", angle_err="")
            rows.append(row)

    summary = [f"Predictions: {pred_lbl_dir.resolve()}",
               f"Truth:       {truth_dir.resolve()}",
               f"Split: {args.split} (holdout = fold{args.fold:02d}test.txt), {len(truth_files)} images, "
               f"{missing_files} with no prediction file",
               f"Predicted boxes: {n_pred} ({n_extra} not matched to any true vehicle)",
               ""]
    summary += summarize(rows, n_truth, "All vehicles")
    for src in ("sam", "corrected", "fallback", "unknown", "unmatched"):
        group = [r for r in rows if r["source"] == src]
        if group:
            summary += [""] + summarize(group, len(group), f"source={src}")

    text = "\n".join(summary)
    print(text)
    tag = f"{args.split}_fold{args.fold:02d}"
    (out_dir / f"evaluation_{tag}.txt").write_text(text + "\n")
    with open(out_dir / f"evaluation_{tag}.csv", "w", newline="") as f:
        fields = ["image", "car", "source", "iou", "true_length", "true_width", "d_length", "d_width", "angle_err"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"\nWrote {out_dir / f'evaluation_{tag}.txt'} and .csv")

if __name__ == "__main__":
    main()
