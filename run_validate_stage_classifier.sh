export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=========================================="
echo "训练分类器"
echo "=========================================="
python examples/validate_stage_classifier.py \
        --exp_name usb_pickup_insertion \
        --ckpt_dir ./stage_classifier_ckpt \
        --num_classes 5 \
        --encoder_type resnet18
