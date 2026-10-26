#!/usr/bin/env python3
"""
Collects evaluation_all_fold01.csv from every <V>/input_dataset_refined_* run
into two tables (sweep_summary.txt: IoU, then length/width errors) and rebuilds mini_compare.html with truth first.
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
    # Size errors (refined minus true, px) over matched boxes: mean absolute, 90th percentile,
    # share within tol px, and mean error relative to the true size
    def size(k, true_k, tol):
        d = np.array([float(r[k]) for r in matched])
        t = np.array([float(r[true_k]) for r in matched])
        if not len(d):
            return {"mae": np.nan, "p90": np.nan, "within": np.nan, "rel": np.nan}
        return {"mae": np.abs(d).mean(), "p90": np.percentile(np.abs(d), 90),
                "within": 100 * (np.abs(d) <= tol).mean(), "rel": 100 * (np.abs(d) / t).mean()}
    sources = {s: sum(r["source"] == s for r in rows) for s in ("sam", "corrected", "fallback", "unmatched")}
    return {"cars": len(rows), "mean": ious.mean(), "median": np.median(ious),
            "ge50": 100 * (ious >= 0.5).mean(), "ge70": 100 * (ious >= 0.7).mean(),
            "d_len": med("d_length"), "d_wid": med("d_width"), "ang": med("angle_err"),
            "matched": len(matched), "len": size("d_length", "true_length", 2),
            "wid": size("d_width", "true_width", 1), **sources}

def main():
    runs = sorted(d for d in V.glob("input_dataset_refined_*") if (d / "evaluation_all_fold01.csv").exists())
    stats = {d.name.replace("input_dataset_refined_", ""): run_stats(d / "evaluation_all_fold01.csv") for d in runs}
    order = sorted(stats, key=lambda n: -stats[n]["mean"])

    w = max(16, max(map(len, order)) + 2)
    header = (f"{'run':<{w}}{'meanIoU':>8}{'median':>8}{'>=0.5':>7}{'>=0.7':>7}"
              f"{'dLen':>7}{'dWid':>7}{'angle':>7}   sam/corr/fallback/unmatched")
    lines = [f"{V}: {stats[order[0]]['cars']} vehicles, sorted by mean IoU", header, "-" * len(header)]
    for n in order:
        s = stats[n]
        lines.append(f"{n:<{w}}{s['mean']:>8.3f}{s['median']:>8.3f}{s['ge50']:>6.0f}%{s['ge70']:>6.0f}%"
                     f"{s['d_len']:>+7.1f}{s['d_wid']:>+7.1f}{s['ang']:>7.1f}   "
                     f"{s['sam']}/{s['corrected']}/{s['fallback']}/{s['unmatched']}")

    # Length/width table, same row order; tolerance 2 px for length, 1 px for width
    header = (f"{'run':<{w}}{'matched':>8} |{'dLen':>6}{'MAE':>6}{'p90':>6}{'<=2px':>7}{'rel':>6} |"
              f"{'dWid':>6}{'MAE':>6}{'p90':>6}{'<=1px':>7}{'rel':>6}")
    lines += ["", f"Length/width errors (refined - true, px; matched boxes): median (dLen/dWid), "
              "mean absolute (MAE), 90th percentile of |error|, share within 2 px (length) / 1 px (width), "
              "mean |error| / true size", header, "-" * len(header)]
    for n in order:
        s = stats[n]
        cols = [f"{s['d_len']:>+6.1f}", f"{s['d_wid']:>+6.1f}"]
        for i, k in enumerate(("len", "wid")):
            z = s[k]
            cols[i] += f"{z['mae']:>6.2f}{z['p90']:>6.1f}{z['within']:>6.0f}%{z['rel']:>5.0f}%"
        lines.append(f"{n:<{w}}{s['matched']:>8} |{cols[0]} |{cols[1]}")
    text = "\n".join(lines)
    print(text)
    (V / "sweep_summary.txt").write_text(text + "\n")

    # Comparison page: truth, then runs from best to worst mean IoU
    cmd = [sys.executable, "data_exploration/compare_plots_html.py", "--run", f"truth={V}/truth"]
    for n in order:
        cmd += ["--run", f"{n} (IoU {stats[n]['mean']:.2f})={V}/input_dataset_refined_{n}"]
    cmd += ["--out", str(V / "mini_compare.html")]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
