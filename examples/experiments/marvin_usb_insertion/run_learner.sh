export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
export WANDB_MODE=disabled && \
python ../../train_rlpd.py "$@" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path=../../experiments/marvin_usb_insertion/checkpoints \
    --demo_path=/home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-07-28_09-47-54.pkl\
    --learner \