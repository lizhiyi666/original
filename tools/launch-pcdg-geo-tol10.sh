#!/usr/bin/env bash
set -euo pipefail
cd /root/experiments/pcdg/pcdg-geo-v1-tol10-20261010
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
exec /root/anaconda3/envs/pcdg-exp/bin/python -u -B tools/run_geometry_study.py \
  --source-run /root/experiments/pcdg/two-city-distance-v2-batched-20261009/experiment_runs/two-city-distance-v2-batched-20261009 \
  --run-id pcdg-geo-v1-tol10-20261010 --device cuda:0 \
  --other-jsd-relative-tolerance 0.10 \
  --inherit-geometry-run /root/experiments/pcdg/pcdg-geo-v1-20261010/experiment_runs/pcdg-geo-v1-20261010 \
  --inherit-geometry-inventory /root/experiments/pcdg/pcdg-geo-v1-tol10-20261010/source-geometry-delivery-inventory.json "$@"
