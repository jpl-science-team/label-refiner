#!/usr/bin/env bash
# Parameter sweep for refine_labels.py on a VEDAI subset made by make_vedai_subset.py
# (a dir with mini_dataset/ and truth/labels). Run from the repo root:
#   bash scripts/run_vedai_sweep.sh                 # VEDAI_mini_test, 12 images, ~10 min
#   V=VEDAI_tune bash scripts/run_vedai_sweep.sh    # whole tune split, ~1 h per setting
#   SWEEP=jitter V=VEDAI_tune_jitterL3W1 bash scripts/run_vedai_sweep.sh   # jittered-point settings
#   SWEEP=holdout V=VEDAI_holdout bash scripts/run_vedai_sweep.sh       # final check, ~10 min per setting
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
    # Two-pass with its checks and --keep-axis-aligned (the correction head rotated real
    # horizontal/vertical cars); listed first so they finish first overnight
    "box_h7_hybrid_keepaxis|--box-half-size 7 --two-pass --keep-axis-aligned"
    "box_h7_hybrid_center_keepaxis|--box-half-size 7 --two-pass --center-on-point --keep-axis-aligned"
    "default|"
    "default_snap|--snap"
    "box_h7|--box-prompt --box-half-size 7"
    "box_h7_snap|--box-prompt --box-half-size 7 --snap"
    "box_h7_dino_snap|--box-prompt --box-half-size 7 --features dino --snap"
    # Centering on the label point raised mean IoU 0.59 -> 0.70 on exact points (offline test on
    # the box_h7 run) but lowered it on jittered points; +2.5 px length helped both by ~0.02
    "box_h7_center|--box-prompt --box-half-size 7 --center-on-point"
    "box_h7_center_grow|--box-prompt --box-half-size 7 --center-on-point --grow-length 2.5"
    # The 14 px box cuts long cars short; --two-pass takes the length from a point-only mask
    # (12-image sets: mean IoU 0.657 exact / 0.654 jitter vs 0.649 / 0.630 for box_h7)
    "box_h7_hybrid|--box-half-size 7 --two-pass"
    "box_h7_hybrid_center|--box-half-size 7 --two-pass --center-on-point"
    "box_h7_keepaxis|--box-prompt --box-half-size 7 --keep-axis-aligned"
  )
elif [ "$SWEEP" = holdout ]; then
  # Final check of the main settings on VEDAI_holdout / VEDAI_holdout_jitterL3W1
  # (make_vedai_subset.py --holdout); don't pick settings from these numbers
  RUNS=(
    "default|"
    "box_h7|--box-prompt --box-half-size 7"
    "box_h7_center|--box-prompt --box-half-size 7 --center-on-point"
    # The correction head rotated real horizontal/vertical cars, and two-pass sometimes took its
    # angle from a nearly square box mask; --keep-axis-aligned and the two-pass checks fix that
    "box_h7_hybrid_keepaxis|--box-half-size 7 --two-pass --keep-axis-aligned"
    "box_h7_hybrid_center_keepaxis|--box-half-size 7 --two-pass --center-on-point --keep-axis-aligned"
  )
else
  echo "Unknown SWEEP=$SWEEP (use boxsize, jitter or holdout)"; exit 1
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
