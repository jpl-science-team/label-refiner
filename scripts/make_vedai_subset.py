#!/usr/bin/env python3
"""
Builds a small VEDAI test set for refine_labels.py sweeps, from the output of
VS_VEDAI_Separation_Conversion.py:

  <out>/input_dataset/{images,labels}   center-point input (copied)
  <out>/truth/labels                   true OBB labels for the same images
  <out>/truth/plots                    truth drawn over the images (for compare_plots_html.py)
  <out>/image_list.txt                 the image names used

Images come from the tune split only (never VEDAI's foldNNtest.txt holdout), unless --holdout
is given, which builds the whole holdout fold instead (for final checks, not tuning). Either pass
--list to rebuild an exact set, or --n-with/--n-empty/--seed to sample one.

VEDAI's points are the exact centers of the true boxes. --jitter R moves each input point by
a random offset of up to R px (uniform over a disc, fixed by --jitter-seed) to simulate
hand-clicked labels; the truth labels are left unchanged. --jitter-length L --jitter-width W
instead moves each point up to L px along its vehicle's length and W px across it (uniform over
an ellipse aligned with the true box), since a click usually lands somewhere along the car.

Examples:
  # The 12-image check set used for the ~12 min sweep (list is checked in)
  python scripts/make_vedai_subset.py --out VEDAI_mini_test --list VEDAI_mini_test/image_list.txt
  # The whole tune split, for an overnight sweep
  python scripts/make_vedai_subset.py --out VEDAI_tune --all
  # The same with points moved up to 3 px along the vehicle and 1 px across it (simulated clicks)
  python scripts/make_vedai_subset.py --out VEDAI_tune_jitterL3W1 --all --jitter-length 3 --jitter-width 1
  # The fold01 holdout (121 images), for final checks only
  python scripts/make_vedai_subset.py --out VEDAI_holdout --holdout
  python scripts/make_vedai_subset.py --out VEDAI_holdout_jitterL3W1 --holdout --jitter-length 3 --jitter-width 1
"""

import argparse
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

IMG_SIZE = 512.0  # VEDAI 512 tiles; labels are normalized to this

def vehicle_axis(obb_line):
    """Unit vector along the long side of a true OBB label (class x1 y1 ... x4 y4, normalized)."""
    pts = [(float(a), float(b)) for a, b in zip(obb_line.split()[1:9:2], obb_line.split()[2:9:2])]
    sides = [(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]) for i in (0, 1)]
    dx, dy = max(sides, key=lambda v: math.hypot(*v))
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n

def jitter_labels(text, radius, rng, truth_text=None, length=0.0, width=0.0):
    """
    Moves each point label (class cx cy w h, normalized), staying inside the image: up to `radius`
    px in any direction, or (with truth_text, one true OBB per point in the same order) up to
    `length` px along the vehicle and `width` px across it.
    """
    truth = [l for l in truth_text.splitlines() if len(l.split()) >= 9] if truth_text is not None else None
    lines = []
    for i, line in enumerate(l for l in text.splitlines() if len(l.split()) >= 5):
        parts = line.split()
        r = math.sqrt(rng.random())  # sqrt: uniform over the disc/ellipse, not bunched at the center
        theta = rng.uniform(0, 2 * math.pi)
        if truth is None:
            dx, dy = radius * r * math.cos(theta), radius * r * math.sin(theta)
        else:
            ux, uy = vehicle_axis(truth[i])
            a, b = length * r * math.cos(theta), width * r * math.sin(theta)
            dx, dy = a * ux - b * uy, a * uy + b * ux
        cx = min(max(float(parts[1]) + dx / IMG_SIZE, 0.0), 1.0)
        cy = min(max(float(parts[2]) + dy / IMG_SIZE, 0.0), 1.0)
        lines.append(f"{parts[0]} {cx:.6f} {cy:.6f} {' '.join(parts[3:])}\n")
    return "".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Build a VEDAI subset with center-point input and truth labels.")
    parser.add_argument("--out", required=True, help="Output dir")
    parser.add_argument("--src", default="datasets/vedai_512/vedai_color_centerpoint", help="Center-point dataset")
    parser.add_argument("--truth", default="datasets/vedai_512/vedai_color_obb/labels", help="True OBB labels")
    parser.add_argument("--folds", default="data/VEDAI/Annotations512", help="Dir with VEDAI's foldNNtest.txt lists")
    parser.add_argument("--fold", type=int, default=1, help="Holdout fold to exclude (default 1)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--list", default=None, help="File of image names (e.g. 00000073_co), one per line")
    which.add_argument("--all", action="store_true", help="Use every tune-split image")
    which.add_argument("--holdout", action="store_true",
                       help="Use every holdout-fold image instead (final checks only; don't tune on it)")
    parser.add_argument("--n-with", type=int, default=11, help="Images with vehicles to sample (default 11)")
    parser.add_argument("--n-empty", type=int, default=1, help="Images without vehicles to sample (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Move each input point by up to this many px (default 0: exact centers)")
    parser.add_argument("--jitter-length", type=float, default=0.0,
                        help="Move each point up to this many px along its vehicle (with --jitter-width)")
    parser.add_argument("--jitter-width", type=float, default=0.0,
                        help="Move each point up to this many px across its vehicle (with --jitter-length)")
    parser.add_argument("--jitter-seed", type=int, default=0)
    args = parser.parse_args()
    directional = args.jitter_length > 0 or args.jitter_width > 0
    if directional and args.jitter > 0:
        parser.error("use either --jitter or --jitter-length/--jitter-width")

    src, truth, out = Path(args.src), Path(args.truth), Path(args.out)
    holdout = set((Path(args.folds) / f"fold{args.fold:02d}test.txt").read_text().split())
    tune = [p.stem for p in sorted((src / "labels").glob("*.txt")) if p.stem.split("_")[0] not in holdout]

    if args.holdout:
        stems = [p.stem for p in sorted((src / "labels").glob("*.txt")) if p.stem.split("_")[0] in holdout]
    elif args.list:
        stems = Path(args.list).read_text().split()
        leaked = [s for s in stems if s.split("_")[0] in holdout]
        if leaked:
            raise ValueError(f"{len(leaked)} images in {args.list} are in the holdout fold, e.g. {leaked[0]}")
    elif args.all:
        stems = tune
    else:
        has_vehicles = lambda s: (src / "labels" / f"{s}.txt").read_text().strip() != ""
        rng = random.Random(args.seed)
        stems = sorted(rng.sample([s for s in tune if has_vehicles(s)], args.n_with)
                       + rng.sample([s for s in tune if not has_vehicles(s)], args.n_empty))

    for d in ("input_dataset/images", "input_dataset/labels", "truth/labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n_vehicles = 0
    jitter_rng = random.Random(args.jitter_seed)
    for s in stems:
        img = next((src / "images").glob(f"{s}.*"))
        shutil.copy(img, out / "input_dataset/images" / img.name)
        points = (src / "labels" / f"{s}.txt").read_text()
        if args.jitter > 0:
            points = jitter_labels(points, args.jitter, jitter_rng)
        elif directional:
            points = jitter_labels(points, 0, jitter_rng, (truth / f"{s}.txt").read_text(),
                                   args.jitter_length, args.jitter_width)
        (out / "input_dataset/labels" / f"{s}.txt").write_text(points)
        shutil.copy(truth / f"{s}.txt", out / "truth/labels" / f"{s}.txt")
        n_vehicles += len((truth / f"{s}.txt").read_text().splitlines())
    if not args.list or Path(args.list).resolve() != (out / "image_list.txt").resolve():
        (out / "image_list.txt").write_text("\n".join(stems) + "\n")

    subprocess.run([sys.executable, "data_exploration/plot_obb_labels.py",
                    "--images", str(out / "input_dataset/images"), "--labels", str(out / "truth/labels"),
                    "--input-labels", str(out / "input_dataset/labels"), "--out", str(out / "truth/plots")],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"{out}: {len(stems)} images, {n_vehicles} vehicles "
          + (f"(holdout fold{args.fold:02d} only)" if args.holdout else f"(holdout fold{args.fold:02d} excluded)")
          + (f", points jittered up to {args.jitter:g} px (seed {args.jitter_seed})" if args.jitter > 0 else "")
          + (f", points jittered up to {args.jitter_length:g} px along / {args.jitter_width:g} px across "
             f"the vehicle (seed {args.jitter_seed})" if directional else ""))

if __name__ == "__main__":
    main()
