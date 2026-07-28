#!/bin/bash
# 清理旧 checkpoint 脚本

echo "=== 清理旧的 checkpoint 数据 ==="
cd /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion

# 创建备份目录
BACKUP_DIR="checkpoints_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "1. 备份到: $BACKUP_DIR"
# 移动旧数据
if [ -d "checkpoints/buffer" ]; then
    mv checkpoints/buffer "$BACKUP_DIR/"
    echo "   ✅ 已备份 buffer/"
fi

if [ -d "checkpoints/demo_buffer" ]; then
    mv checkpoints/demo_buffer "$BACKUP_DIR/"
    echo "   ✅ 已备份 demo_buffer/"
fi

if ls checkpoints/checkpoint_* 1> /dev/null 2>&1; then
    mv checkpoints/checkpoint_* "$BACKUP_DIR/"
    echo "   ✅ 已备份 checkpoint_*"
fi

# 创建新的空目录
echo ""
echo "2. 创建新的空目录"
mkdir -p checkpoints/buffer
mkdir -p checkpoints/demo_buffer

echo ""
echo "✅ 清理完成！"
echo ""
echo "旧数据位置: $BACKUP_DIR"
echo "现在可以重新训练（从 step 0 开始）"
echo ""
echo "如需恢复旧数据："
echo "  mv $BACKUP_DIR/* checkpoints/"
