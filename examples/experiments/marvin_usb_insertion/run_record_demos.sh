#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUCCESS_COUNT="${1:-20}"

if [[ ! "${SUCCESS_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [positive_demo_count]" >&2
    exit 2
fi

echo "[REAL ROBOT] Recording ${SUCCESS_COUNT} successful demos"
echo "[REAL ROBOT] Final condition: scale15, rot0.03, 10 Hz, stream 200 Hz, 60 mm/s"
cd "${EXAMPLES_DIR}"
python record_demos.py \
    --exp_name=marvin_usb_insertion \
    --successes_needed="${SUCCESS_COUNT}"
