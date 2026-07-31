export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9 && \
export WANDB_MODE=disabled && \
python ../../train_rlpd.py "$@" \
    --exp_name=marvin_usb_insertion \
    --checkpoint_path=../../experiments/marvin_usb_insertion/checkpoints_scale10_rot0025_from124k \
    --restore_checkpoint_path=../../experiments/marvin_usb_insertion/checkpoints_scale20_rot005_pre_mdp_migration \
    --restore_checkpoint_step=124000 \
    --migrate_legacy_data \
    --demo_path=/home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-07-28_09-47-54.pkl \
    --learner
