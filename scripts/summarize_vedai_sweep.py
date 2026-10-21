#!/usr/bin/env python3
"""
Collects evaluation_all_fold01.csv from every <V>/mini_dataset_refined_* run
into one table (sweep_summary.txt) and rebuilds mini_compare.html with truth first.
Run from the repo root after scripts/run_vedai_sweep.sh:
  python scripts/summarize_vedai_sweep.py              # VEDAI_mini_test
  V=VEDAI_tune python scripts/summarize_vedai_sweep.py
"""

import csv
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

V = Path(os.environ.get("V", "VEDAI_mini_test"))

def run_stats(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    ious = np.array([float(r["iou"]) for r in rows])
    matched = [r for r in rows if r["d_length"] != ""]
    med = lambda k: np.median([float(r[k]) for r in matched]) if matched else float("nan")
    sources = {s: sum(r["source"] == s for r in rows) for s in ("sam", "corrected", "fallback", "unmatched")}
    return {"cars": len(rows), "mean": ious.mean(), "median": np.median(ious),
            "ge50": 100 * (ious >= 0.5).mean(), "ge70": 100 * (ious >= 0.7).mean(),
            "d_len": med("d_length"), "d_wid": med("d_width"), "ang": med("angle_err"), **sources}

def main():
    runs = sorted(d for d in V.glob("mini_dataset_refined_*") if (d / "evaluation_all_fold01.csv").exists())
    stats = {d.name.replace("mini_dataset_refined_", ""): run_stats(d / "evaluation_all_fold01.csv") for d in runs}
    order = sorted(stats, key=lambda n: -stats[n]["mean"])

    header = (f"{'run':<16}{'meanIoU':>8}{'median':>8}{'>=0.5':>7}{'>=0.7':>7}"
              f"{'dLen':>7}{'dWid':>7}{'angle':>7}   sam/corr/fallback/unmatched")
    lines = [f"{V}: {stats[order[0]]['cars']} vehicles, sorted by mean IoU", header, "-" * len(header)]
    for n in order:
        s = stats[n]
        lines.append(f"{n:<16}{s['mean']:>8.3f}{s['median']:>8.3f}{s['ge50']:>6.0f}%{s['ge70']:>6.0f}%"
                     f"{s['d_len']:>+7.1f}{s['d_wid']:>+7.1f}{s['ang']:>7.1f}   "
                     f"{s['sam']}/{s['corrected']}/{s['fallback']}/{s['unmatched']}")
    text = "\n".join(lines)
    print(text)
    (V / "sweep_summary.txt").write_text(text + "\n")

    # Comparison page: truth, then runs from best to worst mean IoU
    cmd = [sys.executable, "data_exploration/compare_plots_html.py", "--run", f"truth={V}/truth"]
    for n in order:
        cmd += ["--run", f"{n} (IoU {stats[n]['mean']:.2f})={V}/mini_dataset_refined_{n}"]
    cmd += ["--out", str(V / "mini_compare.html")]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
