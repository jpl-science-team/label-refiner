#!/usr/bin/env python3
"""
Draws one run's OBBs and the true OBBs on the same full image, for every image in a VEDAI set
made by make_vedai_subset.py, and writes a page (one image per row, worst mean IoU first).
Truth is white, the run's boxes are green (yellow where the vehicle's IoU < 0.5), and each
vehicle's IoU is written next to it. Tiles are drawn at 2x so 20 px cars are visible.

Example (from the repo root):
  python data_exploration/truth_vs_run_html.py --set VEDAI_holdout \
      --run VEDAI_holdout/input_dataset_refined_box_h7_hybrid_center --out VEDAI_holdout/truth_vs_best.html
View it like compare_plots_html.py's pages (python -m http.server 8765 --bind 127.0.0.1).
"""

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SCALE = 2


def read_obbs(path, w, h):
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) >= 9:
            boxes.append((np.array(v[1:9], float).reshape(4, 2) * [w, h]))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Truth and one run's boxes on the same full image.")
    parser.add_argument("--set", required=True, help="Subset dir from make_vedai_subset.py (input_dataset/, truth/labels)")
    parser.add_argument("--run", required=True, help="refine_labels.py output dir with labels/ (and evaluation_all_fold01.csv)")
    parser.add_argument("--out", required=True, help="HTML page to write; images go to <out stem>_images/")
    args = parser.parse_args()

    vset, run, out = Path(args.set), Path(args.run), Path(args.out)
    img_dir = out.parent / f"{out.stem}_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Per-vehicle IoU, in the same order as the truth label file
    ious = defaultdict(dict)
    eval_csv = run / "evaluation_all_fold01.csv"
    if eval_csv.exists():
        for r in csv.DictReader(open(eval_csv)):
            ious[r["image"]][int(r["car"])] = float(r["iou"])

    rows = []
    for img_path in sorted((vset / "input_dataset/images").glob("*.*")):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        truth = read_obbs(vset / "truth/labels" / f"{stem}.txt", w, h)
        pred = read_obbs(run / "labels" / f"{stem}.txt", w, h)
        car_ious = [ious[stem].get(i) for i in range(len(truth))]

        big = cv2.resize(img, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        for i, box in enumerate(truth):
            cv2.polylines(big, [(box * SCALE).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
        # The run's boxes: match each to the nearest true vehicle to pick its color
        centers = [b.mean(axis=0) for b in truth]
        for box in pred:
            c = box.mean(axis=0)
            iou = None
            if centers:
                j = int(np.argmin([np.linalg.norm(c - t) for t in centers]))
                iou = car_ious[j]
            color = (0, 255, 255) if iou is not None and iou < 0.5 else (0, 255, 0)
            cv2.polylines(big, [(box * SCALE).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        for box, iou in zip(truth, car_ious):
            if iou is None:
                continue
            x, y = (box.max(axis=0) * SCALE).astype(int)
            cv2.putText(big, f"{iou:.2f}", (int(x) + 2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(big, f"{iou:.2f}", (int(x) + 2, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 92])

        known = [v for v in car_ious if v is not None]
        mean = float(np.mean(known)) if known else None
        rows.append((stem, len(truth), len(pred), mean))

    # Worst first; images without vehicles last
    rows.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0))
    parts = [f"<h2>{html.escape(str(run))} vs truth on {html.escape(str(vset))} ({len(rows)} images)</h2>",
             "<p>White: true box. Green: refined box. Yellow: refined box with IoU &lt; 0.5. "
             "Number: that vehicle's IoU. Images are shown at 2&times;, worst mean IoU first.</p>"]
    for stem, n_true, n_pred, mean in rows:
        score = f"mean IoU {mean:.3f}" if mean is not None else "no vehicles"
        parts.append(f"<figure><figcaption><b>{stem}</b>: {n_true} true, {n_pred} refined, {score}</figcaption>"
                     f"<a href='{img_dir.name}/{stem}.jpg'><img src='{img_dir.name}/{stem}.jpg' loading='lazy'></a></figure>")
    out.write_text("<!doctype html><meta charset='utf-8'><title>truth vs run</title>"
                   "<style>body{font-family:sans-serif;background:#222;color:#ddd}"
                   "img{max-width:100%;height:auto}figure{margin:0 0 24px}</style>" + "\n".join(parts))
    print(f"✔ Wrote {len(rows)} images to {out}")


if __name__ == "__main__":
    main()
