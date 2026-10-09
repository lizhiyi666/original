#!/usr/bin/env bash
set -euo pipefail
cd /root/experiments/pcdg/two-city-distance-v2-batched-20261009
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export PYTHONUNBUFFERED=1 WANDB_PROJECT=Marionette WANDB_MODE=online
exec /root/anaconda3/envs/pcdg-exp/bin/python -u -B tools/resume_distance_batched.py \
  --parent-run /root/experiments/pcdg/two-city-distance-v2/experiment_runs/two-city-distance-v2 \
  --run-id two-city-distance-v2-batched-20261009 \
  --delivery-root 'D:/桌面/轨迹/轨迹/轨迹生成/实验/original/experiment_runs/two-city-distance-v2-batched-20261009' "$@"
