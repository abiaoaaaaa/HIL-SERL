export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=/home/xlb/code_marvin/hil-serl/examples/classifier_ckpt/checkpoint_150 \
    --actor \