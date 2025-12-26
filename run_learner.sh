export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
export XLA_PYTHON_CLIENT_ALLOCATOR=platform && \
export PYTHONPATH="$PWD:$PYTHONPATH" && \
python examples/train_rlpd.py "$@" \
    --exp_name=usb_pickup_insertion \
    --checkpoint_path=train_ckpt \
    --demo_path=demo_data/data.pkl \
    --recovery_path=recovery_data/data.pkl \
    --learner \

