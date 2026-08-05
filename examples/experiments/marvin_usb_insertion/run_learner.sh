#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export WANDB_MODE=disabled

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /absolute/path/to/new_demo.pkl [extra train_rlpd arguments...]" >&2
    echo "The demo must be recorded with the final scale15/rot0.03/stream60 configuration." >&2
    exit 2
fi

DEMO_PATH="$1"
shift
RUN_NAME="${RUN_NAME:-checkpoints_scale15_rot003_stream60_256_medium_from392k}"
RUN_DIR="${SCRIPT_DIR}/${RUN_NAME}"
# Warm-start model weights from the previous compatible 256x256 run. Only
# network parameters are restored; optimizer and replay/demo buffers stay new.
PRETRAIN_DIR="${PRETRAIN_DIR:-${SCRIPT_DIR}/checkpoints_final_scale15_rot003_stream60_fresh}"
PRETRAIN_STEP="${PRETRAIN_STEP:-0}"

if [[ ! -f "${DEMO_PATH}" ]]; then
    echo "Demo file not found: ${DEMO_PATH}" >&2
    exit 1
fi
if [[ -d "${RUN_DIR}/buffer" || -d "${RUN_DIR}/demo_buffer" ]] || \
   find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint_*' -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to start a fresh run in non-empty training directory: ${RUN_DIR}" >&2
    echo "Choose another RUN_NAME, or set ALLOW_RESUME=1 only when intentionally resuming." >&2
    if [[ "${ALLOW_RESUME:-0}" != "1" ]]; then
        exit 1
    fi
fi

echo "[learner] output: ${RUN_DIR}"
echo "[learner] fresh demo: ${DEMO_PATH}"
echo "[learner] old online/demo buffers: disabled"

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
    echo "[learner] warm start weights only: ${PRETRAIN_DIR}/checkpoint_${PRETRAIN_STEP}"
else
    echo "[learner] network initialization: fresh (no checkpoint restore)"
fi

VALIDATOR_ARGS=()
if [[ "${ALLOW_UNVERIFIED_DEMO:-0}" == "1" ]]; then
    VALIDATOR_ARGS+=(--allow-missing-metadata)
fi
python "${SCRIPT_DIR}/../../../utils/debug_tools/validate_marvin_demo.py" \
    "${DEMO_PATH}" \
    "${VALIDATOR_ARGS[@]}"

python "${EXAMPLES_DIR}/train_rlpd.py" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path="${RUN_DIR}" \
    --demo_path="${DEMO_PATH}" \
    --learner \
    "${RESTORE_ARGS[@]}" \
    "$@"
