#!/usr/bin/env bash
# Overnight jitter study on the whole VEDAI tune split (1,125 images). Builds two sets
# (exact center points, and points moved up to 4 px), then runs the jitter sweep on both
# at the same time (~7.5 h on 32 CPUs). Run from the repo root inside tmux:
#   tmux new -d -s sweep 'cd ~/label-refiner && conda run -n label-refiner bash scripts/run_vedai_jitter_overnight.sh > VEDAI_jitter_overnight.log 2>&1'
# Restarting skips finished runs. Afterwards:
#   V=VEDAI_tune python scripts/summarize_vedai_sweep.py
#   V=VEDAI_tune_jitter4 python scripts/summarize_vedai_sweep.py
set -e
PY=${PY:-python}

[ -d VEDAI_tune/mini_dataset ] || $PY scripts/make_vedai_subset.py --out VEDAI_tune --all
[ -d VEDAI_tune_jitter4/mini_dataset ] || $PY scripts/make_vedai_subset.py --out VEDAI_tune_jitter4 --all --jitter 4

# Each sweep gets half the cores so they don't slow each other down
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
SWEEP=jitter V=VEDAI_tune bash scripts/run_vedai_sweep.sh > VEDAI_tune_sweep.log 2>&1 &
exact=$!
SWEEP=jitter V=VEDAI_tune_jitter4 bash scripts/run_vedai_sweep.sh > VEDAI_tune_jitter4_sweep.log 2>&1 &
jittered=$!

status=0
wait $exact || { echo "exact-point sweep failed, see VEDAI_tune_sweep.log"; status=1; }
wait $jittered || { echo "jitter sweep failed, see VEDAI_tune_jitter4_sweep.log"; status=1; }

for v in VEDAI_tune VEDAI_tune_jitter4; do
  V=$v $PY scripts/summarize_vedai_sweep.py || status=1
done
echo "overnight jitter study finished (status $status)"
exit $status
