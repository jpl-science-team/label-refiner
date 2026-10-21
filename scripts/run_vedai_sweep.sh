#!/usr/bin/env bash
# Parameter sweep for refine_labels.py on a VEDAI subset made by make_vedai_subset.py
# (a dir with mini_dataset/ and truth/labels). Run from the repo root:
#   bash scripts/run_vedai_sweep.sh                 # VEDAI_mini_test, 12 images, ~10 min
#   V=VEDAI_tune bash scripts/run_vedai_sweep.sh    # whole tune split, ~1 h per setting
#   SWEEP=jitter V=VEDAI_tune_jitter4 bash scripts/run_vedai_sweep.sh   # snap / DINOv2 settings
# Finished runs are skipped, so an interrupted sweep can be restarted.
# Then: python scripts/summarize_vedai_sweep.py  (same V=...) for the table and comparison page.
set -e
PY=${PY:-python}
V=${V:-VEDAI_mini_test}
IN=$V/mini_dataset
CAR_LENGTH=21   # median length of VEDAI car/van/pickup boxes

SWEEP=${SWEEP:-boxsize}

# name | refine_labels.py options
if [ "$SWEEP" = boxsize ]; then
  # On the 12-image check set, --snap, --negative and --features dino + --snap each lowered mean IoU
  # (the VEDAI points are exact box centers), so this sweep focuses on box size. Half-size 5
  # (10 px box) was dropped too: it cut vehicles short (mean IoU 0.43).
  RUNS=(
    "default|"
    "box|--box-prompt"
    "box_h6|--box-prompt --box-half-size 6"
    "box_h7|--box-prompt --box-half-size 7"
    "box_h8|--box-prompt --box-half-size 8"
    "box_h10|--box-prompt --box-half-size 10"
    "box_h14|--box-prompt --box-half-size 14"
  )
elif [ "$SWEEP" = jitter ]; then
  # For off-center points (make_vedai_subset.py --jitter): does snapping the prompt back onto
  # the vehicle help, and do DINOv2 features snap better than SAM's? Run it on both the
  # jittered and the exact-point set to see what jitter costs and what snap wins back.
  RUNS=(
    "default|"
    "default_snap|--snap"
    "box_h7|--box-prompt --box-half-size 7"
    "box_h7_snap|--box-prompt --box-half-size 7 --snap"
    "box_h7_dino_snap|--box-prompt --box-half-size 7 --features dino --snap"
    "box_h8_snap|--box-prompt --box-half-size 8 --snap"
  )
else
  echo "Unknown SWEEP=$SWEEP (use boxsize or jitter)"; exit 1
fi

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
