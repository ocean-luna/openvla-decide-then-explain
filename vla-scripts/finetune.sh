#!/bin/bash
# ============================================================
# Multi-Node Multi-GPU Fine-tuning Script for OpenVLA
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
export VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
: "${DATA_ROOT:?请设置 DATA_ROOT，指向 RLDS 数据集根目录}"
export DATA_ROOT
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-${CODE_DIR}/runs}"
export ADAPTER_TMP_DIR="${ADAPTER_TMP_DIR:-${CODE_DIR}/adapter-tmp}"
export REASONING_DATASET_PATH="${REASONING_DATASET_PATH:-${DATA_ROOT}/libero_reasonings.json}"

# ---------- 训练超参 ----------
export DATASET_NAME=${DATASET_NAME:-"libero_lm_90"}
export LORA_RANK=${LORA_RANK:-32}
export BATCH_SIZE=${BATCH_SIZE:-6}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
export LEARNING_RATE=${LEARNING_RATE:-"8e-4"}
export IMAGE_AUG=${IMAGE_AUG:-False}
export SAVE_LATEST_ONLY=${SAVE_LATEST_ONLY:-False}
export SAVE_STEPS=${SAVE_STEPS:-30000}
export TRAIN_TYPE=${TRAIN_TYPE:-"think_first"}
export MAX_STEPS=${MAX_STEPS:-200000}
export RUN_ID_NOTE="${RUN_ID_NOTE:-${TRAIN_TYPE}}"
# ---------- 环境准备 ----------
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}" || { echo "[ERROR] 无法进入目录: ${CODE_DIR}"; exit 1; }

# ---------- 日志打印 ----------
echo "============================================"
echo "  Multi-Node Multi-GPU Training"
echo "============================================"
echo "  NODE_NUM      : ${NODE_NUM}"
echo "  NODE_RANK     : ${NODE_RANK}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  MASTER_ADDR   : ${MASTER_ADDR}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  CODE_DIR      : ${CODE_DIR}"
echo "  VLA_PATH      : ${VLA_PATH}"
echo "  DATA_ROOT     : ${DATA_ROOT}"
echo "  RUN_ROOT_DIR  : ${RUN_ROOT_DIR}"
echo "  ADAPTER_TMP   : ${ADAPTER_TMP_DIR}"
echo "  DATASET       : ${DATASET_NAME}"
echo "  LORA_RANK     : ${LORA_RANK}"
echo "  BATCH_SIZE    : ${BATCH_SIZE}"
echo "  GRAD_ACCUM    : ${GRAD_ACCUM_STEPS}"
echo "  LEARNING_RATE : ${LEARNING_RATE}"
echo "  IMAGE_AUG     : ${IMAGE_AUG}"
echo "  SAVE_LATEST   : ${SAVE_LATEST_ONLY}"
echo "  SAVE_STEPS    : ${SAVE_STEPS}"
echo "  MAX_STEPS     : ${MAX_STEPS}"
echo "  TRAIN_TYPE    : ${TRAIN_TYPE}"
echo "  RUN_ID_NOTE   : ${RUN_ID_NOTE}"
echo "  REASONING_DATASET_PATH: ${REASONING_DATASET_PATH}"
echo "============================================"

# ---------- 启动训练 ----------
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NODE_NUM}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    vla-scripts/finetune.py \
        --vla_path                    "${VLA_PATH}" \
        --data_root_dir               "${DATA_ROOT}" \
        --dataset_name                "${DATASET_NAME}" \
        --run_root_dir                "${RUN_ROOT_DIR}" \
        --adapter_tmp_dir             "${ADAPTER_TMP_DIR}" \
        --lora_rank                   "${LORA_RANK}" \
        --batch_size                  "${BATCH_SIZE}" \
        --grad_accumulation_steps     "${GRAD_ACCUM_STEPS}" \
        --learning_rate               "${LEARNING_RATE}" \
        --image_aug                   "${IMAGE_AUG}" \
        --save_latest_checkpoint_only "${SAVE_LATEST_ONLY}" \
        --save_steps                  "${SAVE_STEPS}" \
        --max_steps                   "${MAX_STEPS}" \
        --run_id_note                 "${RUN_ID_NOTE}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[ERROR] 训练异常退出，exit code: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo "[INFO] 训练完成！"