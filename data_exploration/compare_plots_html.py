#!/usr/bin/env python3
"""
Builds a static HTML page showing plots from several refinement runs side by side,
one row per image. Pair it with plot_obb_labels.py outputs.

Each --run is NAME=RUN_DIR, where RUN_DIR contains plots/ and labels/.
Per-run box counts are shown under each image: by source (sam / corrected / fallback)
and flagged count if RUN_DIR/refinement_report.csv exists, otherwise total / axis-aligned.

Example:
    python data_exploration/compare_plots_html.py \
        --run original=COWC_mini_test/mini_dataset_refined_original --run current=COWC_mini_test/mini_dataset_refined \
        --out COWC_mini_test/mini_compare.html

View over SSH (from the repo root):
    python -m http.server 8765 --bind 127.0.0.1
    then open http://localhost:8765/COWC_mini_test/mini_compare.html through an SSH tunnel.
    The page polls itself and reloads whenever it is regenerated, so it can stay open.
"""

import argparse
import csv
import html
import os
from pathlib import Path
from collections import Counter
from urllib.parse import quote

import numpy as np


def box_stats(lbl_path):
    if not lbl_path.exists():
        return None
    stats = Counter()
    for line in lbl_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        pts = np.round(np.array(list(map(float, parts[1:9]))).reshape(4, 2), 4)
        stats["boxes"] += 1
        if len(set(pts[:, 0])) <= 2 and len(set(pts[:, 1])) <= 2:
            stats["axis-aligned"] += 1
    return stats


def report_stats(report_path):
    """Per-image Counters of box sources and flags from refinement_report.csv, or None."""
    if not report_path.exists():
        return None
    per_image = {}
    with open(report_path) as f:
        for row in csv.DictReader(f):
            c = per_image.setdefault(Path(row["image"]).stem, Counter())
            c["boxes"] += 1
            c[row["source"]] += 1
            c["flagged"] += int(row["low_similarity"])
    return per_image


def describe(stats, keys):
    if not stats:
        return ""
    return f"{stats['boxes']} boxes: " + ", ".join(f"{stats[k]} {k}" for k in keys if k != "boxes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (repeatable, shown left to right)")
    parser.add_argument("--out", required=True, type=Path, help="Output .html file")
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        name, _, run_dir = spec.partition("=")
        runs.append((name, Path(run_dir)))

    out_dir = args.out.parent.resolve()
    names = sorted({p.name for _, d in runs for p in (d / "plots").glob("*.png")})

    reports = {name: report_stats(run_dir / "refinement_report.csv") for name, run_dir in runs}
    keys = {name: ["sam", "corrected", "fallback", "flagged"] if reports[name] is not None else ["axis-aligned"]
            for name, _ in runs}
    totals = {name: Counter() for name, _ in runs}
    rows = []
    for img_name in names:
        cells = []
        for name, run_dir in runs:
            plot = run_dir / "plots" / img_name
            stem = Path(img_name).stem
            if reports[name] is not None:
                stats = reports[name].get(stem)
            else:
                stats = box_stats(run_dir / "labels" / (stem + ".txt"))
            if stats:
                totals[name].update(stats)
            caption = f"<b>{html.escape(name)}</b>: {describe(stats, keys[name])}"
            if plot.exists():
                # ?v=<mtime> busts the browser cache when a plot is regenerated
                src = quote(os.path.relpath(plot.resolve(), out_dir)) + f"?v={int(plot.stat().st_mtime)}"
                cells.append(f'<figure><a href="{src}" target="_blank"><img src="{src}" loading="lazy"></a>'
                             f"<figcaption>{caption}</figcaption></figure>")
            else:
                cells.append(f"<figure><div class='missing'>missing</div><figcaption>{caption}</figcaption></figure>")
        rows.append(f"<section><h2>{html.escape(img_name)}</h2><div class='row'>{''.join(cells)}</div></section>")

    summary = "<br>".join(f"<b>{html.escape(n)}</b>: {describe(t, keys[n])}" for n, t in totals.items())
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>OBB label comparison</title>
<style>
body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
.row {{ display: grid; grid-template-columns: repeat({len(runs)}, 1fr); gap: 8px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; display: block; border: 1px solid #ccc; }}
figcaption {{ font-size: 13px; padding: 4px 0; }}
h2 {{ font-size: 12px; font-weight: normal; color: #555; word-break: break-all; margin: 24px 0 6px; }}
.missing {{ padding: 40px; text-align: center; color: #999; border: 1px dashed #ccc; }}
</style></head><body>
<h1>OBB label comparison ({len(names)} images)</h1>
<p>Runs with a refinement report: <b style="color:#0a0">green</b> SAM mask accepted &nbsp;
<b style="color:#0aa">cyan</b> replaced by correction head &nbsp; <b style="color:#e80">orange</b> fallback (SAM mask rejected) &nbsp;
<b style="color:#c0c">magenta X</b> low similarity (flagged) &nbsp; <b style="color:#00f">blue dot</b> snapped prompt.<br>
Runs without a report: green = rotated, orange = axis-aligned. &nbsp; <b style="color:#d00">Red dot</b>: original point label.
Click an image to open it full size.</p>
<p>{summary}</p>
{''.join(rows)}
<script>
// Auto-reload when this page is regenerated, so the tab can stay open across runs.
let lastModified = null;
setInterval(async () => {{
  try {{
    const r = await fetch(location.href, {{ method: "HEAD", cache: "no-store" }});
    const lm = r.headers.get("Last-Modified");
    if (lastModified && lm && lm !== lastModified) location.reload();
    lastModified = lm;
  }} catch (e) {{}}
}}, 3000);
</script>
</body></html>
"""
    args.out.write_text(page)
    print(f"✔ Wrote comparison page for {len(names)} images to {args.out}")


if __name__ == "__main__":
    main()
