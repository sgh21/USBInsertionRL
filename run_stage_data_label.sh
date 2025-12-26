export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "=========================================="
echo "打标签 (交互式)"
echo "=========================================="
echo "按 0-3 标记阶段，按 s 保存，按 q 退出"
python -m serl_launcher.utils.stage_data_tool label \
    --dataset_dir ./stage_dataset \
    --num_classes 5 \
    --auto_next