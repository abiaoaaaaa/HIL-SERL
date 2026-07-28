#!/bin/bash
# 内存增长监控脚本

LOG_FILE="memory_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "开始监控内存，日志: $LOG_FILE"
echo "时间,总内存(GB),已用(GB),可用(GB),训练进程RSS(GB)" > $LOG_FILE

while true; do
    TIMESTAMP=$(date +"%H:%M:%S")

    # 系统内存
    MEM_INFO=$(free -g | grep "Mem:")
    TOTAL=$(echo $MEM_INFO | awk '{print $2}')
    USED=$(echo $MEM_INFO | awk '{print $3}')
    AVAILABLE=$(echo $MEM_INFO | awk '{print $7}')

    # 训练进程内存
    PROCESS_RSS=$(ps aux | grep train_rlpd | grep -v grep | awk '{sum+=$6} END {printf "%.1f", sum/1024/1024}')

    # 记录
    echo "$TIMESTAMP,$TOTAL,$USED,$AVAILABLE,$PROCESS_RSS" >> $LOG_FILE

    # 屏幕显示
    clear
    echo "=================================="
    echo "训练内存监控 (Ctrl+C 停止)"
    echo "=================================="
    echo "时间: $TIMESTAMP"
    echo ""
    echo "系统内存:"
    echo "  总计: ${TOTAL}GB"
    echo "  已用: ${USED}GB"
    echo "  可用: ${AVAILABLE}GB"
    echo ""
    echo "训练进程:"
    echo "  RSS: ${PROCESS_RSS}GB"
    echo ""
    echo "日志: $LOG_FILE"

    # 警告
    if (( $(echo "$PROCESS_RSS > 40" | bc -l) )); then
        echo ""
        echo "⚠️  警告: 进程内存 > 40GB!"
    fi

    sleep 5
done
