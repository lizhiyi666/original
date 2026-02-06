#!/bin/bash

# Baseline 2: Post-hoc Swap
# 用法: bash sample_baseline2.sh <run_id>
# 注意: 不使用 --use_constraint_projection，纯扩散采样后做后处理

if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_baseline2.sh <run_id>"
    exit 1
fi

RUN_ID=$1
WORLD_SIZE=4
DATA_NAME="Istanbul_PO1"

echo "====================================================="
echo "Baseline 2: Post-hoc Swap (no projection during sampling)"
echo "Run ID: $RUN_ID, GPUs: $WORLD_SIZE"
echo "====================================================="

for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching GPU $rank ..."
    
    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      --baseline posthoc_swap \
      > "gpu_${rank}_baseline2.log" 2>&1 &
    
    pids[$rank]=$!
done

echo "Waiting for all processes..."
wait
echo "All GPUs done!"

echo "Merging..."
python merge_results.py --run_id "$RUN_ID" --world_size $WORLD_SIZE --data_name "$DATA_NAME"

echo "Evaluating..."
python evaluation.py --datasets "$DATA_NAME" --task Stat --experiment_comments "$RUN_ID"