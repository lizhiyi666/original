#!/bin/bash

if [ $# -ne 1 ]; then
    echo "try: $0 <run_id>, missing your run_id, please check!"
    exit 1
fi

RUN_ID=$1

python sample.py \
  --run_id "$RUN_ID" \
  --use_constraint_projection \
  --debug_constraint_projection \
  --projection_frequency 10 \
  --projection_alm_iters 10 \
  --projection_tau 0.0 \
  --projection_lambda 1.0 \
  --projection_eta 0.2 \
  --projection_mu 1.0

python evaluation.py --datasets Istanbul_PO1 --task Stat --experiment_comments "$RUN_ID" &