#!/bin/bash

# Baseline 3: Classifier-Based Guidance
# 用法: bash sample_baseline3.sh <run_id>

if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_baseline3.sh <run_id>"
    exit 1
fi

RUN_ID=$1
WORLD_SIZE=4
DATA_NAME="Istanbul_PO1"
CLASSIFIER_PATH="./checkpoints/constraint_classifier_${RUN_ID}.pt"
GUIDANCE_SCALE=3.0

# 检查分类器是否已训练
if [ ! -f "$CLASSIFIER_PATH" ]; then
    echo "Classifier not found at $CLASSIFIER_PATH"
    echo "Training classifier first..."
    python train_classifier.py --run_id "$RUN_ID" --epochs 20 --lr 1e-4
fi

echo "====================================================="
echo "Baseline 3: Classifier-Based Guidance"
echo "Run ID: $RUN_ID, Guidance Scale: $GUIDANCE_SCALE"
echo "====================================================="

for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching GPU $rank ..."

    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      --baseline classifier_guidance \
      --classifier_path "$CLASSIFIER_PATH" \
      --guidance_scale $GUIDANCE_SCALE \
      --guidance_last_k_steps 40 \
      --guidance_frequency 4 \
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

echo "====================================================="
echo "Baseline 3 Complete!"
echo "====================================================="