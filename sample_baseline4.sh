#!/bin/bash

# Baseline 4: Classifier-Free Guidance (CFG)
# 用法: bash sample_baseline4.sh <run_id> [cfg_scale]

if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_baseline4.sh <run_id> [cfg_scale]"
    exit 1
fi

RUN_ID=$1
CFG_SCALE=${2:-2.0} # CFG 不需要像 Energy Guidance 那么大的 scale，通常 1.5 - 3.0 即可
WORLD_SIZE=4
DATA_NAME="Istanbul_PO1"

echo "====================================================="
echo "Baseline 4: Classifier-Free Guidance"
echo "Run ID: $RUN_ID, Scale: $CFG_SCALE"
echo "====================================================="

for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching GPU $rank ..."

    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      --baseline cfg \
      --guidance_scale $CFG_SCALE \
      > "gpu_${rank}_baseline4.log" 2>&1 &
done

wait
echo "Merging..."
python merge_results.py --run_id "$RUN_ID" --world_size $WORLD_SIZE --data_name "$DATA_NAME"

echo "Evaluating..."
python evaluation.py --datasets "$DATA_NAME" --task Stat --experiment_comments "$RUN_ID"