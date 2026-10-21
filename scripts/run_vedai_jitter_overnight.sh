#!/usr/bin/env bash
# Overnight jitter study on the whole VEDAI tune split (1,125 images). Builds two sets (exact
# center points, and points moved up to 3 px along the vehicle and 1 px across it), then runs the
# jitter sweep on both at the same time (on 32 CPUs about 2.3 h per setting, 3.3 h for DINOv2). Run from the repo root inside tmux:
#   tmux new -d -s sweep 'cd ~/label-refiner && conda run -n label-refiner bash scripts/run_vedai_jitter_overnight.sh > VEDAI_jitter_overnight.log 2>&1'
# Restarting skips finished runs. Afterwards:
#   V=VEDAI_tune python scripts/summarize_vedai_sweep.py
#   V=VEDAI_tune_jitterL3W1 python scripts/summarize_vedai_sweep.py
set -e
PY=${PY:-python}

[ -d VEDAI_tune/mini_dataset ] || $PY scripts/make_vedai_subset.py --out VEDAI_tune --all
[ -d VEDAI_tune_jitterL3W1/mini_dataset ] || $PY scripts/make_vedai_subset.py --out VEDAI_tune_jitterL3W1 --all \
    --jitter-length 3 --jitter-width 1

SETS=(VEDAI_tune VEDAI_tune_jitterL3W1)
# Each sweep gets an equal share of the cores so they don't slow each other down
export OMP_NUM_THREADS=$(( $(nproc) / ${#SETS[@]} ))
pids=()
for v in "${SETS[@]}"; do
  SWEEP=jitter V=$v bash scripts/run_vedai_sweep.sh > ${v}_sweep.log 2>&1 &
  pids+=($!)
done

status=0
for i in "${!SETS[@]}"; do
  wait ${pids[$i]} || { echo "${SETS[$i]} sweep failed, see ${SETS[$i]}_sweep.log"; status=1; }
done

for v in "${SETS[@]}"; do
  V=$v $PY scripts/summarize_vedai_sweep.py || status=1
done
echo "overnight jitter study finished (status $status)"
exit $status
