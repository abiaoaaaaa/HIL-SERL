export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export WANDB_MODE=disabled && \
python ../../train_rlpd.py "$@" \
      --exp_name=marvin_usb_insertion \
      --checkpoint_path=../../experiments/marvin_usb_insertion/checkpoints_scale10_rot0025_from124k \
      --eval_checkpoint_step=176000 \
      --eval_n_trajs=50 \
      --actor