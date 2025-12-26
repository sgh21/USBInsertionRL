export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=========================================="
echo "提取图像"
echo "=========================================="
python -m serl_launcher.utils.stage_data_tool extract \
    --input_dir ./classifier_data \
    --out_dir ./stage_dataset \
    --key side_stage_classifier \
    # --shuffle