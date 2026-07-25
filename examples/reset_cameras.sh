#!/bin/bash
# RealSense相机重置脚本

echo "================================"
echo "RealSense相机重置工具"
echo "================================"

# 1. 杀死所有相机相关进程
echo "[1/4] 清理相机进程..."
pkill -9 -f "python.*record_success_fail" 2>/dev/null
pkill -9 -f "python.*marvin" 2>/dev/null
pkill -9 -f "debug_classifier" 2>/dev/null
sleep 1

# 2. 找到Intel RealSense USB设备
echo "[2/4] 查找RealSense USB设备..."
lsusb | grep -i "Intel"

# 3. 重置USB设备
echo "[3/4] 重置USB设备..."
for device in /sys/bus/usb/devices/*/product; do
    if grep -qi "RealSense" "$device" 2>/dev/null; then
        parent=$(dirname "$device")
        authorized="$parent/authorized"
        if [ -w "$authorized" ]; then
            echo "  - 重置设备: $parent"
            echo 0 | sudo tee "$authorized" > /dev/null
            sleep 0.5
            echo 1 | sudo tee "$authorized" > /dev/null
            sleep 1
        fi
    fi
done

# 4. 验证相机
echo "[4/4] 验证相机状态..."
timeout 3 rs-enumerate-devices 2>/dev/null || echo "  - rs-enumerate-devices 未安装或超时"

echo ""
echo "✅ 相机重置完成"
echo "现在可以运行数据采集脚本了"
