#!/bin/bash

# Multi-GPU LIBERO Evaluation Script
# Automatically distributes tasks across multiple GPUs

# ================= Configuration =================
# 环境变量优先，命令行参数作为备选
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
CODE_DIR="${CODE_DIR:-${SCRIPT_ROOT}}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}" || exit 1
TASK_SUITE="${TASK_SUITE:-${1:-libero_90}}"
TOTAL_TASKS="${TOTAL_TASKS:-${2:-90}}"
: "${MODEL_PATH:?请设置 MODEL_PATH，指向待评估的论文模型 checkpoint}"
RUN_ID_NOTE="${RUN_ID_NOTE:-${3:-eval}}"
GPU_COUNT="${GPU_COUNT:-8}"
BASE_PORT="${BASE_PORT:-29600}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs/${EVAL_TYPE:-think_first}}"
EVAL_SCRIPT="${EVAL_SCRIPT:-./experiments/robot/libero/run_libero_eval.py}"
EVAL_TYPE="${EVAL_TYPE:-think_first}"  # Options: think_first, answer_first
CENTER_CROP="${CENTER_CROP:-True}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
SEED="${SEED:-7}"
RUN_ID_NOTE="${EVAL_TYPE}--${RUN_ID_NOTE}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-${CODE_DIR}/rollouts/${EVAL_TYPE}}"
ENABLE_GPU_HOLD="${ENABLE_GPU_HOLD:-0}"

# ================= GPU Hold Process Configuration =================
HOLD_GPU_DIR="${TMPDIR:-${CODE_DIR}/.cache}/hold_gpu_${USER:-user}_$$"
HOLD_GPU_SCRIPT="${HOLD_GPU_DIR}/hold_gpu.py"
HOLD_INTERVAL="${HOLD_INTERVAL:-1}"
HOLD_TENSOR_SIZE="${HOLD_TENSOR_SIZE:-1024}"
hold_pids=()

# ================= Create Hold GPU Script =================
create_hold_gpu_script() {
    mkdir -p "$HOLD_GPU_DIR"
    cat > "$HOLD_GPU_SCRIPT" << 'EOF'
import torch
import os
import sys
import time
import signal

# 获取GPU ID
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
tensor_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

# 在 CUDA_VISIBLE_DEVICES 环境下，GPU永远被重新编号为0
device = "cuda:0"

print(f"[GPU Hold] GPU {gpu_id} hold process started (PID: {os.getpid()}, using {device})")
sys.stdout.flush()

# 创建张量占用显存
keep_alive_tensor = torch.randn(tensor_size, tensor_size, device=device)

# 保持运行直到收到终止信号
running = True
def handle_signal(signum, frame):
    global running
    running = False
    print(f"[GPU Hold] GPU {gpu_id} received signal {signum}, shutting down...")
    sys.stdout.flush()

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

try:
    while running:
        # 定期执行轻量计算保持GPU活跃
        result = torch.matmul(keep_alive_tensor, keep_alive_tensor.T)
        result = torch.nn.functional.relu(result)
        time.sleep(interval)
except KeyboardInterrupt:
    pass
finally:
    # 清理显存
    del keep_alive_tensor
    if 'result' in locals():
        del result
    torch.cuda.empty_cache()
    print(f"[GPU Hold] GPU {gpu_id} hold process exited")
    sys.stdout.flush()
EOF
    chmod +x "$HOLD_GPU_SCRIPT"
    echo "[Setup] Created hold GPU script at $HOLD_GPU_SCRIPT"
}

# ================= Cleanup Function =================
cleanup_hold_processes() {
    if [ ${#hold_pids[@]} -gt 0 ]; then
        echo ""
        echo "======================================================"
        echo "Cleaning up ${#hold_pids[@]} GPU hold processes..."
        echo "======================================================"
        for hold_pid in "${hold_pids[@]}"; do
            if kill -0 "$hold_pid" 2>/dev/null; then
                kill -TERM "$hold_pid" 2>/dev/null
                echo "  [Hold] Sent SIGTERM to hold process PID=$hold_pid"
            else
                echo "  [Hold] Hold process PID=$hold_pid already exited"
            fi
        done

        # Wait a bit for graceful shutdown, then force kill if needed
        sleep 5
        for hold_pid in "${hold_pids[@]}"; do
            if kill -0 "$hold_pid" 2>/dev/null; then
                echo "  [Hold] Force killing hold process PID=$hold_pid"
                kill -KILL "$hold_pid" 2>/dev/null
            fi
        done
        echo "[Cleanup] All hold processes cleaned up"
    fi

    # Clean up the temporary directory
    if [ -d "$HOLD_GPU_DIR" ]; then
        rm -rf "$HOLD_GPU_DIR"
    fi
}

# ================= Calculate Task Distribution =================
TASKS_PER_GPU=$((TOTAL_TASKS / GPU_COUNT))
REMAINING=$((TOTAL_TASKS % GPU_COUNT))

echo "======================================================"
echo "Multi-GPU LIBERO Evaluation"
echo "======================================================"
echo "Task suite:    $TASK_SUITE"
echo "Total tasks:   $TOTAL_TASKS"
echo "GPU count:     $GPU_COUNT"
echo "Tasks per GPU: $TASKS_PER_GPU"
echo "Remaining:     $REMAINING"
echo "GPU hold:      ${ENABLE_GPU_HOLD} (0=关闭，1=启用)"
echo "Evaluation:    ${EVAL_TYPE}, trials=${NUM_TRIALS_PER_TASK}, seed=${SEED}"
echo "Log directory: ${LOG_DIR}"
echo "======================================================"

# ================= Initialize Hold GPU Script =================
if [ "${ENABLE_GPU_HOLD}" = "1" ]; then
    create_hold_gpu_script
    trap "cleanup_hold_processes" EXIT
fi

# ================= Launch Evaluation Jobs =================
mkdir -p "$LOG_DIR"

current_start=0
pids=()

for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
    task_end=$((current_start + TASKS_PER_GPU))

    if [ $REMAINING -gt 0 ]; then
        task_end=$((task_end + 1))
        REMAINING=$((REMAINING - 1))
    fi

    if [ $current_start -ge $TOTAL_TASKS ]; then
        echo "GPU $gpu_id: No tasks assigned, skipping"
        continue
    fi

    if [ $task_end -gt $TOTAL_TASKS ]; then
        task_end=$TOTAL_TASKS
    fi

    task_count=$((task_end - current_start))
    # ✅ 每个GPU分配不同端口，避免冲突
    port=$((BASE_PORT + gpu_id))

    echo "Launching GPU $gpu_id: tasks $current_start-$((task_end-1)) ($task_count tasks) port=$port"

    # 🔄 Start hold process for this GPU first
    if [ "${ENABLE_GPU_HOLD}" = "1" ]; then
        (
            export CUDA_VISIBLE_DEVICES=$gpu_id
            export OMP_NUM_THREADS=2
            export MKL_NUM_THREADS=2
            python "$HOLD_GPU_SCRIPT" $gpu_id $HOLD_INTERVAL $HOLD_TENSOR_SIZE
        ) > "$LOG_DIR/${TASK_SUITE}_gpu${gpu_id}_hold.log" 2>&1 &
        hold_pid=$!
        hold_pids+=($hold_pid)
        echo "  [Hold] GPU $gpu_id hold process started (PID: $hold_pid)"
    fi

    # 🚀 Then start the evaluation task
    (
        # ✅ 关键修复：清除所有分布式相关环境变量
        unset WORLD_SIZE
        unset RANK
        unset LOCAL_RANK
        unset MASTER_ADDR
        unset MASTER_PORT

        # ✅ 重新设置为单进程模式
        export WORLD_SIZE=1
        export RANK=0
        export LOCAL_RANK=0
        export MASTER_ADDR="127.0.0.1"
        export MASTER_PORT=$port        # ✅ 每个进程使用唯一端口
        export EVAL_TYPE=$EVAL_TYPE

        # ✅ GPU 与渲染环境隔离
        export CUDA_VISIBLE_DEVICES=$gpu_id

        # ✅ 避免 OMP 线程数过多导致资源竞争
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4

        python "$EVAL_SCRIPT" \
            --model_family openvla \
            --pretrained_checkpoint "$MODEL_PATH" \
            --task_suite_name "$TASK_SUITE" \
            --center_crop "$CENTER_CROP" \
            --local_log_dir "$LOG_DIR" \
            --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
            --seed "$SEED" \
            --task_start $current_start \
            --task_end $task_end \
            --gpu_id $gpu_id \
            --run_id_note "gpu${gpu_id}--${RUN_ID_NOTE}"

    ) > "$LOG_DIR/${TASK_SUITE}_gpu${gpu_id}.log" 2>&1 &

    pids+=($!)
    current_start=$task_end

    # ✅ 稍微错开启动时间，避免同时抢占资源
    sleep 2
done

echo ""
echo "======================================================"
echo "Launched ${#pids[@]} evaluation jobs"
echo "======================================================"
echo ""
echo "Monitor logs:    tail -f $LOG_DIR/${TASK_SUITE}_gpu*.log"
echo "Check running:   ps aux | grep run_libero_eval"
echo "Kill all jobs:   kill ${pids[*]}"
echo ""

# ✅ 等待所有任务并捕获退出状态
echo "Waiting for all jobs to complete..."
failed=0
for pid in "${pids[@]}"; do
    wait "$pid"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "⚠️  Job PID=$pid failed with exit code $exit_code"
        failed=$((failed + 1))
    fi
done

echo ""
echo "======================================================"
if [ $failed -eq 0 ]; then
    echo "✅ All evaluation jobs completed successfully!"
else
    echo "❌ $failed job(s) failed! Check logs for details:"
    echo "   grep -l 'Error\|Traceback' $LOG_DIR/${TASK_SUITE}_gpu*.log"
fi
echo "======================================================"
echo ""
echo "Results:"
echo "  Logs:    $LOG_DIR/${TASK_SUITE}_gpu*.log"
echo "  Videos:  ${ROLLOUT_DIR}"