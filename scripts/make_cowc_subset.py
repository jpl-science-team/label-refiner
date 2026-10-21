#!/usr/bin/env python3
"""
Rebuilds a small COWC test set (flat images/ + labels/) from a list of tile names,
copying them from the output of 00pure_geographic_splitter.py.

Example (the 11-tile COWC mini dataset; list is checked in):
  python scripts/make_cowc_subset.py
  python scripts/make_cowc_subset.py --list COWC_mini_test/image_list.txt --out COWC_mini_test/mini_dataset
"""

import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Copy listed COWC tiles and their point labels into a flat folder.")
    parser.add_argument("--list", default="COWC_mini_test/image_list.txt",
                        help="Tile names (without extension), one per line; '#' lines are comments")
    parser.add_argument("--src", default="datasets/COWC_Points_512",
                        help="Dataset with train|val|test/{images,labels}")
    parser.add_argument("--out", default="COWC_mini_test/mini_dataset", help="Output dir")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    stems = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip() and not l.startswith("#")]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_cars = 0
    for stem in stems:
        # Tile names contain '%' and '.', so match on the exact stem rather than globbing
        matches = [p for split in ("train", "val", "test") for p in (src / split / "images").glob("*")
                   if p.stem == stem]
        if not matches:
            raise FileNotFoundError(f"{stem} not found in {src}/{{train,val,test}}/images")
        img = matches[0]
        lbl = img.parent.parent / "labels" / f"{stem}.txt"
        shutil.copy(img, out / "images" / img.name)
        shutil.copy(lbl, out / "labels" / lbl.name)
        n_cars += sum(1 for l in lbl.read_text().splitlines() if len(l.split()) >= 5)
    print(f"{out}: {len(stems)} tiles, {n_cars} cars")

if __name__ == "__main__":
    main()
