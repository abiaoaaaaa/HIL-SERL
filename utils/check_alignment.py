#!/usr/bin/env python3
"""
观测-动作对齐性检查工具 - Check Observation-Action Alignment

功能：
    验证 demo buffer 中的数据对齐性，检查 obs + action ≈ next_obs 的关系。
    由于动作是增量形式，理论上当前状态加上动作增量应该等于下一个状态。

主要功能：
    1. 从 demo_buffer 加载数据
    2. 随机采样完整的 episode
    3. 对于每个 episode，检查每个 transition 的对齐性
    4. 计算预测状态和实际状态的误差
    5. 生成可视化图表，对比预测值和实际值
    6. 输出统计信息和异常检测

检查内容：
    动作维度 [7]：
        [0-2] dx, dy, dz       平移增量 (m)
        [3-5] drx, dry, drz    旋转增量 (rad)
        [6]   gripper          夹爪命令

    状态维度 [19]：
        [0]      gripper
        [1-3]    force_x/y/z
        [4-9]    tcp_pose (x, y, z, rx, ry, rz)  ← 主要检查目标
        [10-12]  torque_x/y/z
        [13-18]  tcp_vel

    重点检查：
        tcp_pose[t+1] ≈ tcp_pose[t] + action[0:6]
        gripper[t+1] ≈ gripper[t] (或者等于 action[6])

工作流程：
    1. 加载 buffer 数据
    2. 切分完整 episode
    3. 随机采样指定数量的 episode
    4. 对每个 episode 的每个 transition：
       a) 提取当前状态 obs
       b) 提取动作 action
       c) 提取下一个状态 next_obs
       d) 计算预测状态：predicted_next_obs = obs + action
       e) 计算误差：error = next_obs - predicted_next_obs
    5. 生成可视化图表（实际值 vs 预测值）
    6. 输出统计报告

输出格式：
    utils/alignment_check/
    ├── episode_0_alignment.png      # Episode 0 的对齐性图表
    ├── episode_1_alignment.png
    ├── episode_2_alignment.png
    ├── episode_3_alignment.png
    ├── episode_4_alignment.png
    └── alignment_report.txt         # 统计报告

    每个图包含：
    - TCP 位置 (x, y, z) 的实际值 vs 预测值
    - TCP 姿态 (rx, ry, rz) 的实际值 vs 预测值
    - Gripper 的实际值 vs 预测值
    - 误差曲线

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 基本用法（采样 5 个 episode）
    python utils/check_alignment.py

    # 指定采样数量
    python utils/check_alignment.py --num-episodes 10

    # 指定 buffer 文件
    python utils/check_alignment.py \
        --buffer-file examples/experiments/marvin_usb_insertion/checkpoints/buffer/transitions_10000.pkl

    # 指定输出目录
    python utils/check_alignment.py --output-dir ./my_alignment_check

    # 显示详细日志
    python utils/check_alignment.py --verbose

应用场景：
    1. 数据质量检查
    2. 验证动作空间和状态空间的一致性
    3. 发现数据采集或控制中的问题
    4. 调试环境和控制器

注意事项：
    1. 由于控制延迟、动力学等因素，完美对齐是不可能的
    2. 小的误差是正常的，关注异常大的误差
    3. Force、Torque、Velocity 等不受动作直接控制，误差较大是正常的
    4. 主要关注 TCP pose 和 Gripper 的对齐性
"""

import argparse
import glob
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 配置 matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查观测-动作对齐性"
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='examples/experiments/marvin_usb_insertion/checkpoints',
        help='Checkpoints 目录路径'
    )
    parser.add_argument(
        '--buffer-file',
        help='指定 buffer 文件路径（默认：最新文件）'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=5,
        help='采样的 episode 数量（默认：5）'
    )
    parser.add_argument(
        '--output-dir',
        default='utils/alignment_check',
        help='输出目录（默认：utils/alignment_check）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )
    parser.add_argument(
        '--action-scale',
        type=float,
        nargs=3,
        default=[20.0, 0.05, 1.0],
        help='动作缩放参数 [位置(mm) 旋转(rad) 夹爪]（默认：20.0 0.05 1.0）'
    )
    return parser.parse_args()


def load_latest_buffer(checkpoint_dir, buffer_file=None):
    """加载最新的 buffer 文件"""
    if buffer_file:
        if not os.path.exists(buffer_file):
            raise FileNotFoundError(f"Buffer 文件不存在: {buffer_file}")
        return buffer_file

    buffer_dir = os.path.join(checkpoint_dir, 'buffer')
    buffer_files = sorted(
        glob.glob(os.path.join(buffer_dir, 'transitions_*.pkl')),
        key=lambda x: int(x.split('_')[-1].replace('.pkl', ''))
    )

    if not buffer_files:
        raise FileNotFoundError(f"没有找到 buffer 文件: {buffer_dir}")

    return buffer_files[-1]


def is_continuous(t1, t2, atol=1e-6):
    """检查两个 transition 是否连续"""
    next_state = t1['next_observations']['state'].flatten()
    curr_state = t2['observations']['state'].flatten()
    return np.allclose(next_state, curr_state, atol=atol, rtol=1e-5)


def split_episodes(transitions):
    """切分 episodes"""
    episodes = []
    current_episode = [0]

    for i in range(1, len(transitions)):
        t_prev = transitions[i-1]
        t_curr = transitions[i]

        if t_prev['dones'] == 1 or not is_continuous(t_prev, t_curr):
            episodes.append(current_episode)
            current_episode = [i]
        else:
            current_episode.append(i)

    if current_episode:
        episodes.append(current_episode)

    # 只保留完整 episode
    complete_episodes = []
    for ep in episodes:
        last_idx = ep[-1]
        if transitions[last_idx]['dones'] == 1:
            complete_episodes.append(ep)

    return complete_episodes


def check_episode_alignment(transitions, episode_indices, action_scale=None):
    """检查一个 episode 的对齐性

    Args:
        transitions: 所有 transition 数据
        episode_indices: 当前 episode 的索引列表
        action_scale: 动作缩放参数 [位置(mm), 旋转(rad), 夹爪]
                     如果为 None，使用默认值 [20.0, 0.05, 1.0]
    """
    if action_scale is None:
        # 默认值来自 MarvinUSBEnvConfig
        action_scale = np.array([20.0, 0.05, 1.0])

    results = {
        'tcp_pose': {'actual': [], 'predicted': [], 'error': []},
        'gripper': {'actual': [], 'predicted': [], 'error': []},
    }

    for idx in episode_indices:
        t = transitions[idx]

        # 当前状态
        obs = t['observations']['state'].flatten()
        # 动作（归一化，范围 [-1, 1]）
        action = t['actions'].flatten()
        # 下一个状态
        next_obs = t['next_observations']['state'].flatten()

        # 提取关键部分
        # TCP pose: state[4:10] = [x, y, z, rx, ry, rz]（单位：米和弧度）
        tcp_pose_current = obs[4:10]
        tcp_pose_next_actual = next_obs[4:10]

        # 预测下一个 TCP pose
        # action[0:6] = [dx, dy, dz, drx, dry, drz]（归一化值）
        # 需要应用 ACTION_SCALE 转换：
        # - 位置：action[:3] * action_scale[0] (mm) -> 转为米
        # - 旋转：action[3:6] * action_scale[1] (rad)

        pos_delta_m = action[0:3] * action_scale[0] / 1000.0  # mm -> m
        rot_delta_rad = action[3:6] * action_scale[1]  # rad

        action_scaled = np.concatenate([pos_delta_m, rot_delta_rad])
        tcp_pose_next_predicted = tcp_pose_current + action_scaled

        # 误差
        tcp_pose_error = tcp_pose_next_actual - tcp_pose_next_predicted

        # Gripper: state[0]
        gripper_current = obs[0]
        gripper_next_actual = next_obs[0]
        # Gripper 通常保持当前状态，除非有明确的开合命令
        # 这里我们预测为当前状态
        gripper_next_predicted = gripper_current
        gripper_error = gripper_next_actual - gripper_next_predicted

        # 保存结果
        results['tcp_pose']['actual'].append(tcp_pose_next_actual)
        results['tcp_pose']['predicted'].append(tcp_pose_next_predicted)
        results['tcp_pose']['error'].append(tcp_pose_error)

        results['gripper']['actual'].append(gripper_next_actual)
        results['gripper']['predicted'].append(gripper_next_predicted)
        results['gripper']['error'].append(gripper_error)

    # 转换为 numpy 数组
    for key in results:
        for subkey in results[key]:
            results[key][subkey] = np.array(results[key][subkey])

    return results


def plot_alignment(results, episode_id, output_dir):
    """绘制对齐性图表"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f'Episode {episode_id} - Observation-Action Alignment Check',
                 fontsize=14, fontweight='bold')

    steps = np.arange(len(results['tcp_pose']['actual']))

    # TCP pose 标签
    tcp_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']

    # 绘制 TCP pose 的 6 个维度
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        actual = results['tcp_pose']['actual'][:, i]
        predicted = results['tcp_pose']['predicted'][:, i]
        error = results['tcp_pose']['error'][:, i]

        # 绘制实际值和预测值
        ax.plot(steps, actual, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.plot(steps, predicted, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)

        # 绘制误差（在第二个 y 轴）
        ax2 = ax.twinx()
        ax2.plot(steps, error, 'g:', linewidth=1, label='Error', alpha=0.6)
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.set_ylabel('Error', fontsize=8, color='green')
        ax2.tick_params(axis='y', labelcolor='green', labelsize=7)

        # 设置标题和标签
        ax.set_title(f'TCP Pose {tcp_labels[i]}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=7)

        # 计算统计
        mean_error = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        ax.text(0.02, 0.98, f'MAE: {mean_error:.6f}\nMax: {max_error:.6f}',
                transform=ax.transAxes, fontsize=7,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 绘制 Gripper
    ax = axes[3, 0]
    actual = results['gripper']['actual']
    predicted = results['gripper']['predicted']
    error = results['gripper']['error']

    ax.plot(steps, actual, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
    ax.plot(steps, predicted, 'r--', linewidth=1.5, label='Predicted', alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(steps, error, 'g:', linewidth=1, label='Error', alpha=0.6)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_ylabel('Error', fontsize=8, color='green')
    ax2.tick_params(axis='y', labelcolor='green', labelsize=7)

    ax.set_title('Gripper', fontsize=10, fontweight='bold')
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=7)

    mean_error = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    ax.text(0.02, 0.98, f'MAE: {mean_error:.6f}\nMax: {max_error:.6f}',
            transform=ax.transAxes, fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏最后一个空白子图
    axes[3, 1].set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / f'episode_{episode_id}_alignment.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_report(all_results, output_dir, action_scale=None):
    """生成统计报告

    Args:
        all_results: 所有 episode 的检查结果
        output_dir: 输出目录
        action_scale: 动作缩放参数 [位置(mm), 旋转(rad), 夹爪]
    """
    if action_scale is None:
        action_scale = np.array([20.0, 0.05, 1.0])

    report_path = Path(output_dir) / 'alignment_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("观测-动作对齐性检查报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"检查了 {len(all_results)} 个 episodes\n")
        f.write(f"动作缩放: 位置={action_scale[0]:.1f}mm, 旋转={action_scale[1]:.3f}rad, 夹爪={action_scale[2]:.1f}\n\n")

        # TCP Pose 统计
        f.write("-"*80 + "\n")
        f.write("TCP Pose 对齐性统计\n")
        f.write("-"*80 + "\n\n")

        tcp_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Rx (rad)', 'Ry (rad)', 'Rz (rad)']

        for dim_idx, label in enumerate(tcp_labels):
            all_errors = []
            for ep_results in all_results:
                errors = ep_results['tcp_pose']['error'][:, dim_idx]
                all_errors.extend(np.abs(errors))

            all_errors = np.array(all_errors)

            f.write(f"  {label}:\n")
            f.write(f"    Mean Absolute Error: {np.mean(all_errors):.6f}\n")
            f.write(f"    Std Absolute Error:  {np.std(all_errors):.6f}\n")
            f.write(f"    Max Absolute Error:  {np.max(all_errors):.6f}\n")
            f.write(f"    95th Percentile:     {np.percentile(all_errors, 95):.6f}\n")
            f.write("\n")

        # Gripper 统计
        f.write("-"*80 + "\n")
        f.write("Gripper 对齐性统计\n")
        f.write("-"*80 + "\n\n")

        all_gripper_errors = []
        for ep_results in all_results:
            errors = ep_results['gripper']['error']
            all_gripper_errors.extend(np.abs(errors))

        all_gripper_errors = np.array(all_gripper_errors)

        f.write(f"  Mean Absolute Error: {np.mean(all_gripper_errors):.6f}\n")
        f.write(f"  Std Absolute Error:  {np.std(all_gripper_errors):.6f}\n")
        f.write(f"  Max Absolute Error:  {np.max(all_gripper_errors):.6f}\n")
        f.write(f"  95th Percentile:     {np.percentile(all_gripper_errors, 95):.6f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("解读说明\n")
        f.write("="*80 + "\n\n")
        f.write("1. 小的误差（< 0.001）是正常的，来自控制延迟、动力学等因素\n")
        f.write("2. 如果误差持续增大，可能存在数据采集或控制问题\n")
        f.write("3. TCP Pose 的误差应该较小，因为动作直接控制 TCP\n")
        f.write("4. Gripper 的误差取决于具体的控制策略\n")
        f.write("5. 关注 Max Error 和 95th Percentile，识别异常情况\n")

    return report_path


def main():
    args = parse_args()

    print("="*80)
    print("观测-动作对齐性检查工具")
    print("="*80)

    # 转换 action_scale 为 numpy 数组
    action_scale = np.array(args.action_scale)
    print(f"\n动作缩放参数: 位置={action_scale[0]:.1f}mm, 旋转={action_scale[1]:.3f}rad, 夹爪={action_scale[2]:.1f}")

    # 加载 buffer
    buffer_file = load_latest_buffer(args.checkpoint_dir, args.buffer_file)
    print(f"\n加载 buffer: {os.path.basename(buffer_file)}")

    with open(buffer_file, 'rb') as f:
        transitions = pickle.load(f)
    print(f"  总 transitions: {len(transitions)}")

    # 切分 episodes
    print("\n切分 episodes...")
    episodes = split_episodes(transitions)
    print(f"  完整 episode 数: {len(episodes)}")

    if not episodes:
        print("\n错误: 没有找到完整的 episode")
        return 1

    # 采样
    print(f"\n随机采样 {args.num_episodes} 个 episodes...")
    random.seed(args.seed)
    if len(episodes) <= args.num_episodes:
        sampled_episodes = episodes
        print(f"  可用 episode 不足，使用全部 {len(episodes)} 个")
    else:
        sampled_episodes = random.sample(episodes, args.num_episodes)
        print(f"  已采样 {len(sampled_episodes)} 个 episodes")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 检查每个 episode
    print("\n开始检查对齐性...")
    all_results = []

    for ep_id, ep_indices in enumerate(sampled_episodes):
        if args.verbose:
            print(f"  处理 episode {ep_id} (长度: {len(ep_indices)})")

        results = check_episode_alignment(transitions, ep_indices, action_scale=action_scale)
        all_results.append(results)

        output_path = plot_alignment(results, ep_id, output_dir)
        print(f"  已保存: {output_path}")

    # 生成报告
    print("\n生成统计报告...")
    report_path = generate_report(all_results, output_dir, action_scale)
    print(f"  已保存: {report_path}")

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n已检查 {len(sampled_episodes)} 个 episodes")
    print(f"保存位置: {output_dir}")
    print("\n提示:")
    print("  - 蓝色实线：实际的 next_obs")
    print("  - 红色虚线：预测的 next_obs (obs + action)")
    print("  - 绿色点线：误差 (actual - predicted)")
    print("  - 查看 alignment_report.txt 了解统计信息")

    return 0


if __name__ == '__main__':
    sys.exit(main())
