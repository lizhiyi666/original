#!/bin/bash

# 1. 检查参数
if [ -z "$1" ]; then
    echo "Error: Missing Run ID."
    echo "Usage: bash sample_parallel_evaluation.sh <run_id>"
    exit 1
fi

RUN_ID=$1
WORLD_SIZE=4
DATA_NAME="OOD_NewYork_PO1"

echo "====================================================="
echo "Starting Parallel Sampling for Run ID: $RUN_ID"
echo "Using $WORLD_SIZE GPUs simultaneously."
echo "====================================================="

# 2. 并行启动循环
for (( rank=0; rank<$WORLD_SIZE; rank++ ))
do
    echo "Launching process on GPU $rank (Logs -> gpu_${rank}.log)..."
    
    # -----------------------------------------------------------
    #  CUDA_VISIBLE_DEVICES=$rank 指定不同显卡
    #  命令最后面的 '&' 符号，表示后台运行，不阻塞循环
    #  > gpu_${rank}.log 2>&1 将输出重定向，防止屏幕混乱
    # -----------------------------------------------------------
    CUDA_VISIBLE_DEVICES=$rank python sample.py \
      --run_id "$RUN_ID" \
      --rank $rank \
      --world_size $WORLD_SIZE \
      --use_constraint_projection \
      --projection_frequency 2 \
      --projection_outer_iters 15 \
      --projection_inner_iters 25\
      --projection_tau 0 \
      --projection_lambda 0 \
      --projection_eta 0.15 \
      --projection_mu 1.0 \
      --projection_mu_max 1000.0 \
      --projection_mu_alpha 1.3 \
      --projection_delta_tol 0.00000001\
      --use_gumbel_softmax \
      --gumbel_temperature 3.0 \
      --projection_last_k_steps 300 \
      --projection_existence_weight 0.7 \
      > "gpu_${rank}.log" 2>&1 &  
    
    # 保存后台进程 PID (可选，用于调试)
    pids[$rank]=$!
done

# 3. 等待所有任务完成
echo "All processes launched. Waiting for them to finish..."
# 关键点 4: wait 命令挂起脚本，直到上面所有后台 '&' 任务结束
wait 

echo "All GPUs finished sampling!"

# 4. 合并结果
echo "Merging result files..."
python merge_results.py --run_id "$RUN_ID" --world_size $WORLD_SIZE --data_name "$DATA_NAME"

# 5. 评估
echo "Running Evaluation..."
python evaluation.py --datasets "$DATA_NAME" --task Stat --experiment_comments "$RUN_ID"