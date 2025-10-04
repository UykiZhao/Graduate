#!/bin/bash

set -euo pipefail

# Transformer 跨 machine-1 系列堆叠训练示例

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

MACHINES=(
  machine-1-1
  machine-1-2
  machine-1-3
)

INIT_CKPT=""

for MACHINE in "${MACHINES[@]}"; do
  echo "[INFO] Training Transformer on ${MACHINE}"

  RUN_ID="${MACHINE}-${TIMESTAMP}"
  CKPT_TAG="Transformer-${MACHINE}-${TIMESTAMP}.pt"

  CMD=(
    python main.py
    --model Transformer
    --dataset "${MACHINE}"
    --window_length 100
    --d_model 192
    --d_ff 320
    --n_heads 4
    --num_layers 4
    --dropout 0.2
    --batch_size 256
    --learning_rate 5e-4
    --max_epoch 25
    --anomaly_ratio 0.2
    --normalize
    --stride 1
    --run_id "${RUN_ID}"
  )

  if [[ -n "${INIT_CKPT}" ]]; then
    CMD+=(--init_checkpoint "${INIT_CKPT}")
  fi

  "${CMD[@]}"

  if [[ ! -f "${CHECKPOINT_DIR}/Transformer.pt" ]]; then
    echo "[ERROR] Missing checkpoint after training ${MACHINE}" >&2
    exit 1
  fi

  cp "${CHECKPOINT_DIR}/Transformer.pt" "${CHECKPOINT_DIR}/${CKPT_TAG}"
  INIT_CKPT="checkpoints/${CKPT_TAG}"
done

echo "[INFO] Finished stacked training across machine-1-* datasets."
