#!/bin/bash
# ============================================================
# Multi-Node Multi-GPU Training Script for OpenVLA
# ============================================================

# ---------- 分布式环境变量 ----------
export NPROC_PER_NODE=${NPROC_PER_NODE:-${GPU_NUM:-8}}
export NODE_NUM=${NODE_NUM:-1}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29503}

# ---------- 路径变量 ----------
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
export CODE_DIR="${CODE_DIR:-${SCRIPT_ROOT}}"
: "${PRETRAINED_CHECKPOINT:?请设置 PRETRAINED_CHECKPOINT，指向 Prismatic 格式 checkpoint 文件}"
: "${DATA_ROOT:?请设置 DATA_ROOT，指向 RLDS 数据集根目录}"
export PRETRAINED_CHECKPOINT DATA_ROOT
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${CODE_DIR}/runs}"
export REASONING_DATASET_PATH="${REASONING_DATASET_PATH:-${DATA_ROOT}/libero_reasonings.json}"
export TRAIN_TYPE=${TRAIN_TYPE:-"think_first"}

# ---------- 训练配置 ----------
export VLA_TYPE=${VLA_TYPE:-"libero_lm_90"}
export RUN_ID="${RUN_ID:-libero_lm_90_fullytrain--${TRAIN_TYPE}}"
export RESUME_STEP=${RESUME_STEP:-295000}
export RESUME_EPOCH=${RESUME_EPOCH:-40}
export SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
export IMAGE_AUG=${IMAGE_AUG:-True}
export IS_RESUME=${IS_RESUME:-False}
DEFAULT_TRACKERS='["tensorboard"]'
export TRACKERS="${TRACKERS:-${DEFAULT_TRACKERS}}"
export LEARNING_RATE=${LEARNING_RATE:-3e-5}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
export EPOCHS=${EPOCHS:-5}
# ---------- 环境准备 ----------
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"
export TF_CPP_MIN_LOG_LEVEL=3

# 计算 world_size 并设置环境变量
WORLD_SIZE=$((NODE_NUM * NPROC_PER_NODE))
export EXPECTED_WORLD_SIZE=$WORLD_SIZE
GLOBAL_BATCH_SIZE=$((EXPECTED_WORLD_SIZE * PER_DEVICE_BATCH_SIZE))
export GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE
cd "${CODE_DIR}" || { echo "[ERROR] 无法进入目录: ${CODE_DIR}"; exit 1; }

# ---------- 日志打印 ----------
echo "============================================"
echo "  Multi-Node Multi-GPU Training"
echo "============================================"
echo "  NODE_NUM      : ${NODE_NUM}"
echo "  NODE_RANK     : ${NODE_RANK}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  WORLD_SIZE    : ${WORLD_SIZE}"
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  CODE_DIR      : ${CODE_DIR}"
echo "  PRETRAINED_CKPT: ${PRETRAINED_CHECKPOINT}"
echo "  DATA_ROOT     : ${DATA_ROOT}"
echo "  RUN_ROOT_DIR  : ${RUN_ROOT_DIR}"
echo "  VLA_TYPE      : ${VLA_TYPE}"
echo "  RUN_ID        : ${RUN_ID}"
echo "  RESUME        : ${IS_RESUME}"
echo "  RESUME_STEP   : ${RESUME_STEP}"
echo "  RESUME_EPOCH  : ${RESUME_EPOCH}"
echo "  SAVE_INTERVAL : ${SAVE_INTERVAL}"
echo "  IMAGE_AUG     : ${IMAGE_AUG}"
echo "  TRACKERS      : ${TRACKERS}"
echo "  LEARNING_RATE : ${LEARNING_RATE}"
echo "  PER_DEVICE_BATCH_SIZE : ${PER_DEVICE_BATCH_SIZE}"
echo "  EPOCHS        : ${EPOCHS}"
echo "  MAX_STEPS     : ${MAX_STEPS:-未设置，使用 epochs}"
echo "  REASONING_DATASET_PATH: ${REASONING_DATASET_PATH}"
echo "  TRAIN_TYPE    : ${TRAIN_TYPE}"
echo "============================================"
echo ""
echo "LIBERO_VLA Environment Variables:"
echo "  EXPECTED_WORLD_SIZE=$EXPECTED_WORLD_SIZE"
echo "  GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
echo ""

# 构建训练命令
TORCHRUN_ARGS=(
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes="${NODE_NUM}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
)

TRAIN_ARGS=(
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
    --vla.type "${VLA_TYPE}"
    --data_root_dir "${DATA_ROOT}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --run_id "${RUN_ID}"
    --image_aug "${IMAGE_AUG}"
    --save_interval "${SAVE_INTERVAL}"
    --is_resume "${IS_RESUME}"
    --trackers "${TRACKERS}"
)

# 添加 resume 参数
if [ "${IS_RESUME}" = "True" ]; then
    TRAIN_ARGS+=(--resume_step "${RESUME_STEP}" --resume_epoch "${RESUME_EPOCH}")
fi

# ---------- 启动训练 ----------
torchrun "${TORCHRUN_ARGS[@]}" vla-scripts/train.py "${TRAIN_ARGS[@]}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] 训练异常退出，exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo "[INFO] 训练完成！"