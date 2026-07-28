#!/bin/bash
# Marvin USB Insertion 脚踏板配置脚本
# 需要先安装 footswitch: sudo make install

echo "="
echo "配置脚踏板..."
echo "="

# 检查 footswitch 是否安装
if ! command -v footswitch &> /dev/null; then
    echo "❌ 错误: footswitch 未安装"
    echo "请先运行:"
    echo "  sudo apt-get install libhidapi-dev"
    echo "  git clone https://github.com/rgerganov/footswitch.git"
    echo "  cd footswitch && make && sudo make install"
    exit 1
fi

# 修复权限
echo "修复设备权限..."
sudo chmod 666 /dev/hidraw10 /dev/hidraw11 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 权限已修复"
else
    echo "⚠️  权限修复失败，尝试使用 sudo 配置"
fi

# 配置踏板
echo "正在配置踏板..."

# 踏板1 = Shift + 左方向键（标记成功）
sudo footswitch -1 -m shift -k left
if [ $? -eq 0 ]; then
    echo "✅ 踏板1 → Shift+← (标记成功)"
else
    echo "❌ 踏板1 配置失败"
    exit 1
fi

# 踏板3 = Shift + 右方向键（触发重置）
sudo footswitch -3 -m shift -k right
if [ $? -eq 0 ]; then
    echo "✅ 踏板3 → Shift+→ (触发重置)"
else
    echo "❌ 踏板3 配置失败"
    exit 1
fi

echo ""
echo "="
echo "配置完成！验证配置："
sudo footswitch -r
echo ""
echo "现在可以："
echo "  🦶 踩踏板1 (左) = 标记成功"
echo "  🦶 踩踏板3 (右) = 触发重置"
echo "="
