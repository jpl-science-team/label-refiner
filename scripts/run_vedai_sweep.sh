#!/usr/bin/env bash
# Parameter sweep for refine_labels.py on a VEDAI subset made by make_vedai_subset.py
# (a dir with mini_dataset/ and truth/labels). Run from the repo root:
#   bash scripts/run_vedai_sweep.sh                 # VEDAI_mini_test, 12 images, ~10 min
#   V=VEDAI_tune bash scripts/run_vedai_sweep.sh    # whole tune split, ~1 h per setting
# Finished runs are skipped, so an interrupted sweep can be restarted.
# Then: python scripts/summarize_vedai_sweep.py  (same V=...) for the table and comparison page.
set -e
PY=${PY:-python}
V=${V:-VEDAI_mini_test}
IN=$V/mini_dataset
CAR_LENGTH=21   # median length of VEDAI car/van/pickup boxes

# name | refine_labels.py options
# On the 12-image check set, --snap, --negative and --features dino + --snap each lowered mean IoU
# (the VEDAI points are exact box centers), so the sweep now focuses on box size.
RUNS=(
  "default|"
  "box|--box-prompt"
  "box_h5|--box-prompt --box-half-size 5"
  "box_h6|--box-prompt --box-half-size 6"
  "box_h7|--box-prompt --box-half-size 7"
  "box_h8|--box-prompt --box-half-size 8"
  "box_h10|--box-prompt --box-half-size 10"
  "box_h14|--box-prompt --box-half-size 14"
)

start=$SECONDS
for entry in "${RUNS[@]}"; do
  name=${entry%%|*}
  opts=${entry#*|}
  out=$V/mini_dataset_refined_$name
  if [ ! -f "$out/refinement_report.csv" ]; then
    t=$SECONDS
    echo "=== $name: $opts"
    $PY scripts/refine_labels.py --input $IN --output $out --car-length $CAR_LENGTH $opts 2>&1 \
        | grep -E "Car length|Reference|Boxes"
    echo "    $((SECONDS - t)) s"
  fi
  $PY scripts/evaluate_updated_vedai_labels.py --pred $out --truth $V/truth/labels --split all > /dev/null
  $PY data_exploration/plot_obb_labels.py --images $IN/images --labels $out/labels \
      --input-labels $IN/labels --out $out/plots > /dev/null 2>&1
done
echo "sweep done in $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s"
