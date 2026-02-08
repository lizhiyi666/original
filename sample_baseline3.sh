#!/bin/bash

# Baseline 3: Classifier-Based Guidance
# 用法: bash sample_baseline3.sh <run_id> [guidance_scale] [guidance_temperature]

if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_baseline3.sh <run_id> [scale] [temperature]"
    exit 1
fi

RUN_ID=$1
GUIDANCE_SCALE=${2:-10.0}
GUIDANCE_TEMP=${3:-1.0}
WORLD_SIZE=4
DATA_NAME="Istanbul_PO1"

echo "====================================================="
echo "Baseline 3: Classifier-Based Guidance"
echo "Run ID: $RUN_ID, Scale: $GUIDANCE_SCALE, Temp: $GUIDANCE_TEMP"
echo "====================================================="

for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching GPU $rank ..."

    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      --baseline energy_guidance \
      --guidance_scale $GUIDANCE_SCALE \
      --guidance_temperature $GUIDANCE_TEMP \
      --guidance_last_k_steps 40 \
      --guidance_frequency 4 \
      --projection_existence_weight 5 \
      > "gpu_${rank}_baseline3.log" 2>&1 &

    pids[$rank]=$!
done

echo "Waiting for all processes..."
wait
echo "All GPUs done!"

echo "Merging..."
python merge_results.py --run_id "$RUN_ID" --world_size $WORLD_SIZE --data_name "$DATA_NAME"

echo "Evaluating..."
python evaluation.py --datasets "$DATA_NAME" --task Stat --experiment_comments "$RUN_ID"