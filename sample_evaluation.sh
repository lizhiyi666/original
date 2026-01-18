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
  --projection_frequency 1 \
  --projection_outer_iters 100 \
  --projection_inner_iters 100\
  --projection_tau 0 \
  --projection_lambda 0.0 \
  --projection_eta 0.2 \
  --projection_mu 1.0 \
  --projection_mu_max 1000.0 \
  --projection_mu_alpha 2.0 \
  --projection_delta_tol 0.000001\
  --use_gumbel_softmax \
  --gumbel_temperature 0.1 \
  --projection_last_k_steps 100 \
  --projection_existence_weight 5.0 \

python evaluation.py --datasets Istanbul_PO1 --task Stat --experiment_comments "$RUN_ID" &


