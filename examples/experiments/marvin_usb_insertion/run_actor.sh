export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_MODE=disabled && \
python ../../train_rlpd.py "$@" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path=../../experiments/marvin_usb_insertion/checkpoints \
    --actor \