#!/bin/bash
# 训练前释放内存脚本

echo "=========================================="
echo "训练前内存清理"
echo "=========================================="

echo "1. 当前内存状态:"
free -h

echo -e "\n2. 关闭 Chrome..."
pkill -9 chrome 2>/dev/null || echo "Chrome 未运行"

echo -e "\n3. 清理系统缓存..."
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo -e "\n4. 清理后内存状态:"
free -h

echo -e "\n5. 最占内存的进程 (Top 10):"
ps aux --sort=-%mem | head -11

echo -e "\n=========================================="
echo "内存清理完成，可以开始训练"
echo "=========================================="
