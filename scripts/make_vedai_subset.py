#!/usr/bin/env python3
"""
Builds a small VEDAI test set for refine_labels.py sweeps, from the output of
VS_VEDAI_Separation_Conversion.py:

  <out>/mini_dataset/{images,labels}   center-point input (copied)
  <out>/truth/labels                   true OBB labels for the same images
  <out>/truth/plots                    truth drawn over the images (for compare_plots_html.py)
  <out>/image_list.txt                 the image names used

Images come from the tune split only (never VEDAI's foldNNtest.txt holdout). Either pass
--list to rebuild an exact set, or --n-with/--n-empty/--seed to sample one.

Examples:
  # The 12-image check set used for the ~12 min sweep (list is checked in)
  python scripts/make_vedai_subset.py --out VEDAI_mini_test --list VEDAI_mini_test/image_list.txt
  # The whole tune split, for an overnight sweep
  python scripts/make_vedai_subset.py --out VEDAI_tune --all
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Build a VEDAI subset with center-point input and truth labels.")
    parser.add_argument("--out", required=True, help="Output dir")
    parser.add_argument("--src", default="datasets/vedai_512/vedai_color_centerpoint", help="Center-point dataset")
    parser.add_argument("--truth", default="datasets/vedai_512/vedai_color_obb/labels", help="True OBB labels")
    parser.add_argument("--folds", default="data/VEDAI/Annotations512", help="Dir with VEDAI's foldNNtest.txt lists")
    parser.add_argument("--fold", type=int, default=1, help="Holdout fold to exclude (default 1)")
    parser.add_argument("--list", default=None, help="File of image names (e.g. 00000073_co), one per line")
    parser.add_argument("--all", action="store_true", help="Use every tune-split image")
    parser.add_argument("--n-with", type=int, default=11, help="Images with vehicles to sample (default 11)")
    parser.add_argument("--n-empty", type=int, default=1, help="Images without vehicles to sample (default 1)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src, truth, out = Path(args.src), Path(args.truth), Path(args.out)
    holdout = set((Path(args.folds) / f"fold{args.fold:02d}test.txt").read_text().split())
    tune = [p.stem for p in sorted((src / "labels").glob("*.txt")) if p.stem.split("_")[0] not in holdout]

    if args.list:
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

    for d in ("mini_dataset/images", "mini_dataset/labels", "truth/labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n_vehicles = 0
    for s in stems:
        img = next((src / "images").glob(f"{s}.*"))
        shutil.copy(img, out / "mini_dataset/images" / img.name)
        shutil.copy(src / "labels" / f"{s}.txt", out / "mini_dataset/labels" / f"{s}.txt")
        shutil.copy(truth / f"{s}.txt", out / "truth/labels" / f"{s}.txt")
        n_vehicles += len((truth / f"{s}.txt").read_text().splitlines())
    if not args.list or Path(args.list).resolve() != (out / "image_list.txt").resolve():
        (out / "image_list.txt").write_text("\n".join(stems) + "\n")

    subprocess.run([sys.executable, "data_exploration/plot_obb_labels.py",
                    "--images", str(out / "mini_dataset/images"), "--labels", str(out / "truth/labels"),
                    "--input-labels", str(out / "mini_dataset/labels"), "--out", str(out / "truth/plots")],
                   check=True, stdout=subprocess.DEVNULL)
    print(f"{out}: {len(stems)} images, {n_vehicles} vehicles (holdout fold{args.fold:02d} excluded)")

if __name__ == "__main__":
    main()
