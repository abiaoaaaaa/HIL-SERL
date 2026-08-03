#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export WANDB_MODE=disabled

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints_final_scale15_rot003_stream60_fresh}"
CHECKPOINT_STEP="${1:-latest}"

if [[ "${CHECKPOINT_STEP}" == "latest" ]]; then
    LATEST_CHECKPOINT="$(
        find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'checkpoint_*' \
            -printf '%f\n' | sort -V | tail -n 1
    )"
    if [[ -z "${LATEST_CHECKPOINT}" ]]; then
        echo "No checkpoint found in ${CHECKPOINT_DIR}" >&2
        exit 1
    fi
    CHECKPOINT_STEP="${LATEST_CHECKPOINT##*_}"
elif [[ ! "${CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [checkpoint_step|latest]" >&2
    exit 2
fi

if [[ ! -d "${CHECKPOINT_DIR}/checkpoint_${CHECKPOINT_STEP}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_DIR}/checkpoint_${CHECKPOINT_STEP}" >&2
    exit 1
fi

EVAL_N_TRAJS="${EVAL_N_TRAJS:-30}"

echo "[REAL ROBOT] Evaluating checkpoint_${CHECKPOINT_STEP} for ${EVAL_N_TRAJS} episodes"
echo "[REAL ROBOT] Marvin controller: 192.168.14.190"

python "${EXAMPLES_DIR}/train_rlpd.py" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path="${CHECKPOINT_DIR}" \
    --eval_checkpoint_step="${CHECKPOINT_STEP}" \
    --eval_n_trajs="${EVAL_N_TRAJS}" \
    --actor
