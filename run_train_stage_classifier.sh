export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=========================================="
echo "训练分类器"
echo "=========================================="
python ./examples/train_stage_classifier.py \
    --exp_name usb_pickup_insertion \
    --dataset_dir ./stage_dataset \
    --num_classes 5 \
    --num_epochs 150 \
    --batch_size 64 \
    --lr 1e-4 \
    --encoder_type resnet18 \
    --save_best

echo "=========================================="
echo "完成！检查点保存在 stage_classifier_ckpt/"
echo "=========================================="