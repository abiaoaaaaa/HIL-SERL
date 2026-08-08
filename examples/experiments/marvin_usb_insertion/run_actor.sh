#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export WANDB_MODE=disabled

RUN_NAME="${RUN_NAME:-checkpoints_256_medium_newbuffer_v2}"
RUN_DIR="${SCRIPT_DIR}/${RUN_NAME}"
PRETRAIN_DIR="${PRETRAIN_DIR-${SCRIPT_DIR}/checkpoints_scale15_rot003_stream60_256_medium_from392k}"
PRETRAIN_STEP="${PRETRAIN_STEP:-0}"

RESTORE_ARGS=()
if [[ -n "${PRETRAIN_DIR}" ]]; then
    if [[ "${PRETRAIN_STEP}" == "0" ]]; then
        LATEST_CHECKPOINT="$(
            find "${PRETRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint_*' \
                -printf '%f\n' | sort -V | tail -n 1
        )"
        if [[ -z "${LATEST_CHECKPOINT}" ]]; then
            echo "No checkpoint found in ${PRETRAIN_DIR}" >&2
            exit 1
        fi
        PRETRAIN_STEP="${LATEST_CHECKPOINT##*_}"
    fi
    if [[ ! -d "${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}" ]]; then
        echo "Checkpoint not found: ${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}" >&2
        exit 1
    fi
    RESTORE_ARGS+=(
        --restore_checkpoint_path="${PRETRAIN_DIR}"
        --restore_checkpoint_step="${PRETRAIN_STEP}"
        --restore_weights_only
    )
    echo "[actor] warm start weights only: ${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}"
else
    echo "[actor] network initialization: fresh; waiting for learner parameters"
fi

echo "[actor] output: ${RUN_DIR}"

python "${EXAMPLES_DIR}/train_rlpd.py" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path="${RUN_DIR}" \
    "${RESTORE_ARGS[@]}" \
    --ip=localhost \
    --actor \
    "$@"
