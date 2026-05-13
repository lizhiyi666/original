#!/bin/bash

# Baseline 1: Vanilla Marionette (原始模型，无任何投影与干预)
# 用法: bash sample_baseline1.sh <run_id> [data_name]
# 示例: bash sample_baseline1.sh marionette_vanilla_ood Istanbul_PO1_OOD

if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_baseline1.sh <run_id> [data_name]"
    exit 1
fi

RUN_ID=$1
# 默认使用 OOD 数据集，如果你想测普通数据集，可以在执行时传入第二个参数
DATA_NAME=${2:-"OOD_NewYork_PO1"} 
WORLD_SIZE=4

echo "====================================================="
echo "Baseline 1: Vanilla Marionette (Pure Native Sampling)"
echo "Run ID: $RUN_ID, GPUs: $WORLD_SIZE, Dataset: $DATA_NAME"
echo "====================================================="

for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching GPU $rank ..."

    # 【关键】这里没有任何 --baseline 或 --use_constraint_projection 标志
    # 完全依赖原始模型的原生能力进行采样
    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      > "gpu_${rank}_baseline1.log" 2>&1 &
done

echo "Waiting for all sampling processes to finish..."
wait
echo "All GPUs done!"

echo "Merging results..."
python merge_results.py --run_id "$RUN_ID" --world_size $WORLD_SIZE --data_name "$DATA_NAME"

echo "Evaluating Baseline 1..."
python evaluation.py --datasets "$DATA_NAME" --task Stat --experiment_comments "${RUN_ID}"