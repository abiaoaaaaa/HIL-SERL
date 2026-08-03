#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export WANDB_MODE=disabled

RUN_NAME="${RUN_NAME:-checkpoints_final_scale15_rot003_stream60_fresh}"
RUN_DIR="${SCRIPT_DIR}/${RUN_NAME}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${SCRIPT_DIR}/checkpoints_scale10_rot0025_from124k}"
PRETRAIN_STEP="${PRETRAIN_STEP:-266000}"

if [[ ! -d "${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}" ]]; then
    echo "Pretrained checkpoint not found: ${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}" >&2
    exit 1
fi

echo "[actor] output: ${RUN_DIR}"
echo "[actor] warm start: ${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}"

python "${EXAMPLES_DIR}/train_rlpd.py" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path="${RUN_DIR}" \
    --restore_checkpoint_path="${PRETRAIN_DIR}" \
    --restore_checkpoint_step="${PRETRAIN_STEP}" \
    --restore_weights_only \
    --ip=localhost \
    --actor \
    "$@"
