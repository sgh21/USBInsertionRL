export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
export PYTHONPATH="$PWD:$PYTHONPATH" && \
python examples/train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=train_ckpt \
    --actor \
