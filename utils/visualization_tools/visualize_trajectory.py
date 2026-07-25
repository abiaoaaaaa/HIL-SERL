#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹数据可视化工具 - Visualize Trajectory Data

功能：
    读取 debug_step_timing.py 或 debug_movla_spacemouse.py 生成的
    轨迹数据 JSON 文件，生成多种可视化图表用于分析和对比。

主要功能：
    1. 关节角度轨迹对比图（10Hz vs 20Hz）
    2. 时间分析图（movLA、发送、总时间）
    3. 发送时间戳分析图（检测抖动）
    4. 支持多文件对比分析

生成的图表：

    1. 关节角度轨迹对比图（joint_trajectories_comparison.png）：
       - 6 个子图，每个对应一个关节
       - 蓝色线：10Hz 控制频率的轨迹
       - 橙色线：20Hz 控制频率的轨迹
       - X 轴：时间戳（相对于开始时间）
       - Y 轴：关节角度（弧度）
       - 用途：对比不同频率下的轨迹平滑度

    2. 时间分析图（timing_analysis.png）：
       包含 4 个子图：
       a) movLA 规划时间：
          - 每个 step 的规划耗时（毫秒）
          - 检测规划时间的稳定性
       b) 轨迹点发送时间：
          - 每个 step 的总发送耗时（毫秒）
          - 与轨迹点数的关系
       c) Step 总执行时间 vs 控制周期：
          - 每个 step 的实际耗时
          - 与理论控制周期的对比
          - 检测超时情况
       d) 规划点数变化：
          - 每个 step 规划的轨迹点数量
          - 点数变化趋势

    3. 发送时间戳分析图（send_timestamps.png）：
       - 散点图：每个轨迹点的发送耗时
       - X 轴：step 编号
       - Y 轴：发送耗时（毫秒）
       - 颜色：按 step 分组
       - 用途：检测发送过程中的抖动和异常

工作流程：
    1. 加载数据：
       - 读取两个 JSON 文件（10Hz 和 20Hz）
       - 解析配置和 steps 数据
       - 过滤掉 skipped 的 steps

    2. 数据预处理：
       - 提取时间戳并归一化（相对于开始时间）
       - 提取关节角度序列
       - 提取时序信息（movLA、发送、总时间）
       - 提取轨迹点数信息

    3. 生成图表：
       - 使用 matplotlib 绘图
       - 设置中文字体（支持中文标签）
       - 自动调整布局
       - 保存为 PNG 文件

    4. 输出结果：
       - 保存到指定输出目录
       - 打印文件路径

数据要求：
    输入 JSON 文件格式：
    {
        "config": {
            "control_hz": 10 或 20,
            "movla_freq_hz": 500,
            ...
        },
        "steps": [
            {
                "step_id": 0,
                "timestamp_start": 时间戳,
                "timestamp_end": 时间戳,
                "joints_before": [7个关节角度],
                "joints_after": [7个关节角度],
                "planned_points": [[7], [7], ...],
                "num_planned_points": 点数,
                "time_movla": 秒,
                "time_send_total": 秒,
                "send_timestamps": [时间列表],
                "total_time": 秒
            },
            ...
        ]
    }

输出文件：
    <output_dir>/
    ├── joint_trajectories_comparison.png  - 关节轨迹对比
    ├── timing_analysis.png               - 时间分析
    └── send_timestamps.png               - 发送时间戳分析

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 基本用法（对比 10Hz 和 20Hz）
    python utils/visualization_tools/visualize_trajectory.py \
        utils/debug_tools/trajectory_data_10hz_20240125.json \
        utils/debug_tools/trajectory_data_20hz_20240125.json

    # 指定输出目录
    python utils/visualization_tools/visualize_trajectory.py \
        utils/debug_tools/trajectory_data_10hz_20240125.json \
        utils/debug_tools/trajectory_data_20hz_20240125.json \
        --output ./trajectory_analysis

    # 使用通配符（自动匹配最新文件）
    python utils/visualization_tools/visualize_trajectory.py \
        utils/debug_tools/trajectory_data_10hz_*.json \
        utils/debug_tools/trajectory_data_20hz_*.json

参数说明：
    位置参数：
    - file_10hz: 10Hz 控制频率的数据文件路径
    - file_20hz: 20Hz 控制频率的数据文件路径

    可选参数：
    - --output: 输出目录（默认: ./trajectory_analysis）

分析建议：

    1. 关节轨迹图：
       - 观察轨迹是否平滑
       - 对比 10Hz 和 20Hz 的差异
       - 检测是否有抖动或振荡
       - 关注关节 3、4、5（通常更敏感）

    2. 时间分析图：
       - movLA 时间应该相对稳定
       - 发送时间应该与点数成正比
       - 总时间不应频繁超过控制周期
       - 点数变化应该平缓

    3. 发送时间戳图：
       - 点应该均匀分布
       - 避免出现明显的分层或跳跃
       - 检测是否有周期性抖动

典型工作流程：
    1. 使用 debug_step_timing.py 记录数据：
       python utils/debug_tools/debug_step_timing.py --hz 10
       python utils/debug_tools/debug_step_timing.py --hz 20

    2. 文本分析（快速查看）：
       python utils/visualization_tools/analyze_trajectory_data.py \
           utils/debug_tools/trajectory_data_10hz_*.json \
           utils/debug_tools/trajectory_data_20hz_*.json

    3. 可视化分析（深入对比）：
       python utils/visualization_tools/visualize_trajectory.py \
           utils/debug_tools/trajectory_data_10hz_*.json \
           utils/debug_tools/trajectory_data_20hz_*.json

    4. 查看图表：
       open trajectory_analysis/*.png

应用场景：
    1. 对比不同控制频率的性能
    2. 诊断轨迹抖动问题
    3. 优化 movLA 参数
    4. 验证轨迹规划效果
    5. 生成报告和演示材料

注意事项：
    1. 两个输入文件应该是相同任务的不同频率
    2. 确保数据文件完整且格式正确
    3. 需要安装 matplotlib：pip install matplotlib
    4. 中文标签可能需要配置中文字体
    5. 图表文件会覆盖同名文件
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_data(filename):
    """加载轨迹数据"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_joint_trajectories(data_10hz, data_20hz, output_dir):
    """绘制关节角度轨迹对比"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('关节角度轨迹对比 (10Hz vs 20Hz)', fontsize=16)

    # 6个关节
    for joint_idx in range(6):
        row = joint_idx // 3
        col = joint_idx % 3
        ax = axes[row, col]

        # 10Hz 数据
        if data_10hz["steps"]:
            timestamps_10 = []
            joints_10 = []
            for step in data_10hz["steps"]:
                if "skipped" not in step:
                    t = step["timestamp_start"]
                    timestamps_10.append(t)
                    joints_10.append(step["joints_before"][joint_idx])

                    # 添加 after 点
                    timestamps_10.append(step["timestamp_end"])
                    joints_10.append(step["joints_after"][joint_idx])

            if timestamps_10:
                t0_10 = timestamps_10[0]
                timestamps_10 = [(t - t0_10) for t in timestamps_10]
                ax.plot(timestamps_10, joints_10, 'b-', alpha=0.7, linewidth=1, label='10Hz')

        # 20Hz 数据
        if data_20hz["steps"]:
            timestamps_20 = []
            joints_20 = []
            for step in data_20hz["steps"]:
                if "skipped" not in step:
                    t = step["timestamp_start"]
                    timestamps_20.append(t)
                    joints_20.append(step["joints_before"][joint_idx])

                    # 添加 after 点
                    timestamps_20.append(step["timestamp_end"])
                    joints_20.append(step["joints_after"][joint_idx])

            if timestamps_20:
                t0_20 = timestamps_20[0]
                timestamps_20 = [(t - t0_20) for t in timestamps_20]
                ax.plot(timestamps_20, joints_20, 'r-', alpha=0.7, linewidth=1, label='20Hz')

        ax.set_xlabel('时间 (s)')
        ax.set_ylabel(f'关节 {joint_idx+1} 角度 (deg)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'joint_trajectories_comparison.png', dpi=150)
    print(f"[保存] 关节轨迹对比图: {output_dir / 'joint_trajectories_comparison.png'}")
    plt.close()


def plot_timing_analysis(data_10hz, data_20hz, output_dir):
    """绘制时间分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('时间分析 (10Hz vs 20Hz)', fontsize=16)

    # 提取时间数据
    def extract_times(data):
        movla_times = []
        send_times = []
        total_times = []
        num_points = []

        for step in data["steps"]:
            if "skipped" not in step:
                movla_times.append(step["time_movla"] * 1000)  # ms
                send_times.append(step["time_send_total"] * 1000)  # ms
                total_times.append(step["total_time"] * 1000)  # ms
                num_points.append(step["num_planned_points"])

        return movla_times, send_times, total_times, num_points

    movla_10, send_10, total_10, points_10 = extract_times(data_10hz)
    movla_20, send_20, total_20, points_20 = extract_times(data_20hz)

    # 1. movLA 规划时间
    ax = axes[0, 0]
    ax.plot(movla_10, 'b-', alpha=0.7, label=f'10Hz (avg={np.mean(movla_10):.2f}ms)')
    ax.plot(movla_20, 'r-', alpha=0.7, label=f'20Hz (avg={np.mean(movla_20):.2f}ms)')
    ax.set_xlabel('Step')
    ax.set_ylabel('movLA 时间 (ms)')
    ax.set_title('movLA 规划时间')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 发送时间
    ax = axes[0, 1]
    ax.plot(send_10, 'b-', alpha=0.7, label=f'10Hz (avg={np.mean(send_10):.2f}ms)')
    ax.plot(send_20, 'r-', alpha=0.7, label=f'20Hz (avg={np.mean(send_20):.2f}ms)')
    ax.set_xlabel('Step')
    ax.set_ylabel('发送时间 (ms)')
    ax.set_title('轨迹点发送时间')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 总时间
    ax = axes[1, 0]
    control_period_10 = 1000 / 10  # 100ms
    control_period_20 = 1000 / 20  # 50ms
    ax.plot(total_10, 'b-', alpha=0.7, label=f'10Hz (avg={np.mean(total_10):.2f}ms)')
    ax.axhline(control_period_10, color='b', linestyle='--', alpha=0.5, label=f'10Hz 周期 ({control_period_10:.0f}ms)')
    ax.plot(total_20, 'r-', alpha=0.7, label=f'20Hz (avg={np.mean(total_20):.2f}ms)')
    ax.axhline(control_period_20, color='r', linestyle='--', alpha=0.5, label=f'20Hz 周期 ({control_period_20:.0f}ms)')
    ax.set_xlabel('Step')
    ax.set_ylabel('总时间 (ms)')
    ax.set_title('Step 总执行时间 vs 控制周期')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 规划点数
    ax = axes[1, 1]
    ax.plot(points_10, 'b-', alpha=0.7, label=f'10Hz (avg={np.mean(points_10):.1f}点)')
    ax.plot(points_20, 'r-', alpha=0.7, label=f'20Hz (avg={np.mean(points_20):.1f}点)')
    ax.set_xlabel('Step')
    ax.set_ylabel('点数')
    ax.set_title('movLA 规划的轨迹点数')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'timing_analysis.png', dpi=150)
    print(f"[保存] 时间分析图: {output_dir / 'timing_analysis.png'}")
    plt.close()


def plot_send_timestamps(data_10hz, data_20hz, output_dir):
    """绘制发送时间戳分析（关键：看是否连续）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('轨迹点发送时间戳分析（检测抖动）', fontsize=16)

    # 10Hz
    ax = axes[0]
    step_idx = 0
    for step in data_10hz["steps"]:
        if "skipped" not in step and "send_timestamps" in step:
            timestamps = np.array(step["send_timestamps"]) * 1000  # ms
            x = np.arange(len(timestamps)) + step_idx * 100
            ax.scatter(x, timestamps, s=10, alpha=0.6)
            step_idx += 1

    ax.set_xlabel('轨迹点索引')
    ax.set_ylabel('单点发送时间 (ms)')
    ax.set_title(f'10Hz - 每个点的发送耗时')
    ax.grid(True, alpha=0.3)

    # 20Hz
    ax = axes[1]
    step_idx = 0
    for step in data_20hz["steps"]:
        if "skipped" not in step and "send_timestamps" in step:
            timestamps = np.array(step["send_timestamps"]) * 1000  # ms
            x = np.arange(len(timestamps)) + step_idx * 100
            ax.scatter(x, timestamps, s=10, alpha=0.6, color='red')
            step_idx += 1

    ax.set_xlabel('轨迹点索引')
    ax.set_ylabel('单点发送时间 (ms)')
    ax.set_title(f'20Hz - 每个点的发送耗时')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'send_timestamps.png', dpi=150)
    print(f"[保存] 发送时间戳分析: {output_dir / 'send_timestamps.png'}")
    plt.close()


def print_statistics(data_10hz, data_20hz):
    """打印统计信息"""
    print("\n" + "="*70)
    print("统计信息")
    print("="*70)

    for data, label in [(data_10hz, "10Hz"), (data_20hz, "20Hz")]:
        print(f"\n【{label}】")
        print(f"  总 step 数: {len(data['steps'])}")

        movla_times = []
        send_times = []
        total_times = []
        num_points = []

        for step in data["steps"]:
            if "skipped" not in step:
                movla_times.append(step["time_movla"] * 1000)
                send_times.append(step["time_send_total"] * 1000)
                total_times.append(step["total_time"] * 1000)
                num_points.append(step["num_planned_points"])

        if movla_times:
            control_period = 1000 / data["config"]["control_hz"]
            print(f"  控制周期: {control_period:.1f} ms")
            print(f"  平均 movLA 时间: {np.mean(movla_times):.2f} ms (±{np.std(movla_times):.2f})")
            print(f"  平均发送时间: {np.mean(send_times):.2f} ms (±{np.std(send_times):.2f})")
            print(f"  平均总时间: {np.mean(total_times):.2f} ms (±{np.std(total_times):.2f})")
            print(f"  平均规划点数: {np.mean(num_points):.1f} (±{np.std(num_points):.1f})")

            # 检查是否超时
            timeout_count = sum(1 for t in total_times if t > control_period)
            print(f"  超时 step 数: {timeout_count} / {len(total_times)} ({timeout_count/len(total_times)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='轨迹数据可视化')
    parser.add_argument('file_10hz', help='10Hz 数据文件')
    parser.add_argument('file_20hz', help='20Hz 数据文件')
    parser.add_argument('--output', default='./trajectory_analysis', help='输出目录')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # 加载数据
    print(f"[加载] 10Hz 数据: {args.file_10hz}")
    data_10hz = load_data(args.file_10hz)

    print(f"[加载] 20Hz 数据: {args.file_20hz}")
    data_20hz = load_data(args.file_20hz)

    # 打印统计信息
    print_statistics(data_10hz, data_20hz)

    # 生成图表
    print("\n[生成] 绘制图表...")
    plot_joint_trajectories(data_10hz, data_20hz, output_dir)
    plot_timing_analysis(data_10hz, data_20hz, output_dir)
    plot_send_timestamps(data_10hz, data_20hz, output_dir)

    print(f"\n[完成] 所有图表已保存到: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
