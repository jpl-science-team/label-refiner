#!/usr/bin/env python3
"""
Zoomed side-by-side crops of every true vehicle, one row per vehicle and one column per run,
with the run's box (and the true box, thin white) drawn on an enlarged crop. Small vehicles
(~20 px on VEDAI 512) are hard to judge on full-tile plots; this makes size errors visible.

Each --run is NAME=RUN_DIR (a refine_labels.py output with labels/; the per-car IoU comes from
RUN_DIR/evaluation_all_fold01.csv when present). Rows are sorted by the first run's IoU, worst first.

Example (from the repo root):
  python data_exploration/vehicle_crops_html.py --set VEDAI_mini_test \
      --run box_h7=VEDAI_mini_test/input_dataset_refined_box_h7 \
      --run two_pass=VEDAI_mini_test/input_dataset_refined_box_h7_hybrid \
      --out VEDAI_mini_test/crops.html
View it like compare_plots_html.py's pages (python -m http.server 8765 --bind 127.0.0.1).
"""

import argparse
import csv
import html
import time
import cv2
import numpy as np
from pathlib import Path

CROP = 48    # crop size around each true vehicle center (px)
ZOOM = 5     # enlargement factor for display

def load_obbs(path, w, h):
    if not path.exists():
        return []
    rows = [l.split() for l in path.read_text().splitlines() if len(l.split()) >= 9]
    return [np.array(r[1:9], dtype=float).reshape(4, 2) * [w, h] for r in rows]

def load_ious(run_dir):
    csv_path = run_dir / "evaluation_all_fold01.csv"
    if not csv_path.exists():
        return {}
    return {(r["image"], int(r["car"])): float(r["iou"]) for r in csv.DictReader(open(csv_path))}

def nearest(boxes, c):
    if not boxes:
        return None
    return min(boxes, key=lambda b: np.linalg.norm(b.mean(axis=0) - c))

def draw_crop(img, c, truth_box, pred_box, point):
    x0, y0 = int(round(c[0])) - CROP // 2, int(round(c[1])) - CROP // 2
    pad = cv2.copyMakeBorder(img, CROP, CROP, CROP, CROP, cv2.BORDER_CONSTANT)
    crop = pad[y0 + CROP:y0 + 2 * CROP, x0 + CROP:x0 + 2 * CROP]
    big = cv2.resize(crop, (CROP * ZOOM, CROP * ZOOM), interpolation=cv2.INTER_NEAREST)
    to_big = lambda b: ((b - [x0, y0]) * ZOOM).astype(np.int32)
    cv2.polylines(big, [to_big(truth_box)], True, (255, 255, 255), 1, cv2.LINE_AA)
    if pred_box is not None:
        cv2.polylines(big, [to_big(pred_box)], True, (0, 255, 0), 2, cv2.LINE_AA)
    if point is not None:
        cv2.circle(big, tuple(to_big(np.array([point]))[0]), 4, (0, 0, 255), -1)
    return big

def main():
    parser = argparse.ArgumentParser(description="Zoomed per-vehicle crops comparing refinement runs.")
    parser.add_argument("--set", required=True, help="Subset dir from make_vedai_subset.py (input_dataset/, truth/labels)")
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (repeatable, shown left to right)")
    parser.add_argument("--out", required=True, help="HTML page to write; crops go to <out stem>_crops/")
    args = parser.parse_args()

    vset, out = Path(args.set), Path(args.out)
    runs = [(n, Path(d)) for n, d in (r.split("=", 1) for r in args.run)]
    crop_dir = out.parent / f"{out.stem}_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    ious = {n: load_ious(d) for n, d in runs}

    rows = []
    for truth_path in sorted((vset / "truth/labels").glob("*.txt")):
        stem = truth_path.stem
        img = cv2.imread(str(next((vset / "input_dataset/images").glob(f"{stem}.*"))))
        h, w = img.shape[:2]
        truth = load_obbs(truth_path, w, h)
        pts_rows = [l.split() for l in (vset / "input_dataset/labels" / f"{stem}.txt").read_text().splitlines() if l.strip()]
        points = [np.array([float(r[1]) * w, float(r[2]) * h]) for r in pts_rows]
        preds = {n: load_obbs(d / "labels" / f"{stem}.txt", w, h) for n, d in runs}
        for i, t in enumerate(truth):
            c = t.mean(axis=0)
            point = nearest(points, c)
            cells = []
            for n, _ in runs:
                name = f"{stem}_{i}_{n}.png"
                cv2.imwrite(str(crop_dir / name), draw_crop(img, c, t, nearest(preds[n], c), point))
                cells.append((name, ious[n].get((stem, i))))
            rows.append((stem, i, cells))
    first = runs[0][0]
    rows.sort(key=lambda r: r[2][0][1] if r[2][0][1] is not None else 1.0)

    v = int(time.time())
    css = ("body{background:#111;color:#ddd;font-family:sans-serif} table{border-collapse:collapse}"
           "td,th{padding:4px;text-align:center;font-size:13px} img{display:block}")
    head = "".join(f"<th>{html.escape(n)}</th>" for n, _ in runs)
    body = []
    for stem, i, cells in rows:
        tds = "".join(f'<td><img src="{crop_dir.name}/{name}?v={v}" width="{CROP * ZOOM}">'
                      f'IoU {iou:.2f}</td>' if iou is not None else
                      f'<td><img src="{crop_dir.name}/{name}?v={v}" width="{CROP * ZOOM}"></td>'
                      for name, iou in cells)
        body.append(f"<tr><td>{stem}<br>car {i}</td>{tds}</tr>")
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(str(vset))} crops</title>
<style>{css}</style></head><body>
<p>{html.escape(str(vset))}: {len(rows)} vehicles, worst {html.escape(first)} IoU first.
White = true box, green = run's box, red dot = input point. Crops are {CROP} px, shown {ZOOM}x.</p>
<table><tr><th></th>{head}</tr>{''.join(body)}</table></body></html>
""")
    print(f"✔ Wrote {len(rows)} vehicle rows to {out}")

if __name__ == "__main__":
    main()
