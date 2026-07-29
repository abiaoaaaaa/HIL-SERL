#!/usr/bin/env python3
"""
Demo 轨迹完整可视化工具 - Visualize Demo Actions and States

功能：
    从 demo_buffer 中随机采样若干条完整轨迹，在同一张图上绘制动作和状态。
    用不同颜色区分人工干预和机器人自主执行的部分。

主要功能：
    1. 从 demo_buffer 加载数据
    2. 按 dones 和连续性切分 episode
    3. 随机采样指定数量的完整轨迹
    4. 区分人工干预和机器人执行
    5. 为每条轨迹生成一张完整图（包含 6 个动作子图 + 19 个状态子图）
    6. 用颜色区分：人工=蓝色，机器人=橙色

输出格式：
    utils/demo_full_plots/
    ├── trajectory_0_full.png    # 第 1 条轨迹（动作+状态）
    ├── trajectory_1_full.png    # 第 2 条轨迹
    ├── trajectory_2_full.png
    ├── trajectory_3_full.png
    └── trajectory_4_full.png

    每个图包含 25 个子图（5 行 × 5 列）：

    ┌──────────────────────── 动作空间 (6个) ───────────────────────┐
    │ dx       │ dy       │ dz       │ drx      │ dry      │
    │ (Trans X)│ (Trans Y)│ (Trans Z)│ (Roll)   │ (Pitch)  │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ drz      │          │          │          │          │
    │ (Yaw)    │ (分隔)   │ (分隔)   │ (分隔)   │ (分隔)   │
    ├──────────┴──────────┴──────────┴──────────┴──────────┤
    │                    状态空间 (19个)                     │
    ├──────────┬──────────┬──────────┬──────────┬──────────┤
    │ Gripper  │ Force X  │ Force Y  │ Force Z  │ TCP X    │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ TCP Y    │ TCP Z    │ TCP Rx   │ TCP Ry   │ TCP Rz   │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ Torque X │ Torque Y │ Torque Z │ Vel X    │ Vel Y    │
    ├──────────┼──────────┼──────────┼──────────┼──────────┤
    │ Vel Z    │ Vel Rx   │ Vel Ry   │ Vel Rz   │ (空)     │
    └──────────┴──────────┴──────────┴──────────┴──────────┘

    颜色说明：
    - 蓝色：人工干预（demo buffer 中的数据）
    - 橙色：机器人执行（仅在 buffer 中的数据）

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 基本用法（采样 5 条轨迹）
    python utils/visualize_demo_full.py

    # 指定采样数量
    python utils/visualize_demo_full.py --num-trajectories 10

    # 指定输出目录
    python utils/visualize_demo_full.py --output-dir ./my_full_plots

    # 使用中文标签（需要中文字体）
    python utils/visualize_demo_full.py --use-chinese

应用场景：
    1. 全面了解轨迹的动作和状态
    2. 对比动作指令和实际状态的关系
    3. 一张图看清整个执行过程
    4. 生成报告和演示材料

注意事项：
    1. 需要同时有 buffer 和 demo_buffer 文件
    2. 图像较大，适合高分辨率显示
    3. 包含 6 个动作维度 + 19 个状态维度
    4. 需要安装 matplotlib：pip install matplotlib
"""

import argparse
import glob
import hashlib
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 配置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 动作维度名称（6维）
ACTION_NAMES = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
ACTION_LABELS = [
    'dx (Translation X)',
    'dy (Translation Y)',
    'dz (Translation Z)',
    'drx (Roll)',
    'dry (Pitch)',
    'drz (Yaw)'
]

# 状态维度名称（19维）
STATE_NAMES = [
    'Gripper',
    'Force X', 'Force Y', 'Force Z',
    'TCP X', 'TCP Y', 'TCP Z',
    'TCP Rx', 'TCP Ry', 'TCP Rz',
    'Torque X', 'Torque Y', 'Torque Z',
    'Vel X', 'Vel Y', 'Vel Z',
    'Vel Rx', 'Vel Ry', 'Vel Rz',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="可视化 demo buffer 中的轨迹动作和状态，区分人工和机器人执行"
    )
    parser.add_argument(
        '--checkpoint-dir',
        default='examples/experiments/marvin_usb_insertion/checkpoints1',
        help='Checkpoints 目录路径'
    )
    parser.add_argument(
        '--buffer-file',
        help='指定 buffer 文件路径（默认：最新文件）'
    )
    parser.add_argument(
        '--demo-file',
        help='指定 demo_buffer 文件路径（默认：最新文件）'
    )
    parser.add_argument(
        '--num-trajectories',
        type=int,
        default=5,
        help='采样的轨迹数量（默认：5）'
    )
    parser.add_argument(
        '--output-dir',
        default='utils/demo_full_plots',
        help='输出目录（默认：utils/demo_full_plots）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )
    parser.add_argument(
        '--use-chinese',
        action='store_true',
        help='使用中文标签（需要系统安装中文字体）'
    )
    return parser.parse_args()


def transition_signature(transition):
    """计算 transition 的唯一签名"""
    digest = hashlib.blake2b(digest_size=16)
    state = np.asarray(transition['observations']['state']).reshape(-1)
    action = np.asarray(transition['actions']).reshape(-1)
    next_state = np.asarray(transition['next_observations']['state']).reshape(-1)

    for arr in [state, action, next_state]:
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.dtype).encode('ascii'))
        digest.update(str(arr.shape).encode('ascii'))
        digest.update(arr.tobytes())

    return digest.digest()


def load_latest_files(checkpoint_dir):
    """加载最新的 buffer 和 demo_buffer 文件"""
    buffer_dir = os.path.join(checkpoint_dir, 'buffer')
    demo_dir = os.path.join(checkpoint_dir, 'demo_buffer')

    buffer_files = sorted(
        glob.glob(os.path.join(buffer_dir, 'transitions_*.pkl')),
        key=lambda x: int(x.split('_')[-1].replace('.pkl', ''))
    )

    if not buffer_files:
        raise FileNotFoundError(f"没有找到 buffer 文件: {buffer_dir}")

    buffer_file = buffer_files[-1]
    demo_file = os.path.join(demo_dir, os.path.basename(buffer_file))

    if not os.path.exists(demo_file):
        raise FileNotFoundError(f"没有找到对应的 demo_buffer 文件: {demo_file}")

    return buffer_file, demo_file


def load_and_mark_human(buffer_file, demo_file):
    """加载数据并标记人工演示"""
    print(f"\n加载数据...")
    print(f"Buffer:      {os.path.basename(buffer_file)}")
    print(f"Demo buffer: {os.path.basename(demo_file)}")

    with open(buffer_file, 'rb') as f:
        transitions = pickle.load(f)
    print(f"  Buffer transitions: {len(transitions)}")

    with open(demo_file, 'rb') as f:
        demo_transitions = pickle.load(f)
    print(f"  Demo transitions:   {len(demo_transitions)}")

    print("\n计算签名...")
    demo_signatures = set(transition_signature(t) for t in demo_transitions)

    is_human = np.array([
        transition_signature(t) in demo_signatures
        for t in transitions
    ], dtype=bool)

    print(f"  人工演示: {is_human.sum()} / {len(transitions)} "
          f"({100*is_human.mean():.1f}%)")

    return transitions, is_human


def is_continuous(t1, t2, atol=1e-6):
    """检查两个 transition 是否连续"""
    next_state = t1['next_observations']['state'].flatten()
    curr_state = t2['observations']['state'].flatten()
    return np.allclose(next_state, curr_state, atol=atol, rtol=1e-5)


def split_episodes(transitions, is_human):
    """切分 episodes"""
    print("\n切分 episodes...")

    episodes = []
    current_episode = {'indices': [0], 'is_human': [is_human[0]]}

    for i in range(1, len(transitions)):
        t_prev = transitions[i-1]
        t_curr = transitions[i]

        if t_prev['dones'] == 1 or not is_continuous(t_prev, t_curr):
            episodes.append(current_episode)
            current_episode = {'indices': [i], 'is_human': [is_human[i]]}
        else:
            current_episode['indices'].append(i)
            current_episode['is_human'].append(is_human[i])

    if current_episode['indices']:
        episodes.append(current_episode)

    complete_episodes = []
    for ep in episodes:
        last_idx = ep['indices'][-1]
        if transitions[last_idx]['dones'] == 1:
            complete_episodes.append(ep)

    print(f"  总 episode 数: {len(episodes)}")
    print(f"  完整 episode 数: {len(complete_episodes)}")

    if complete_episodes:
        lengths = [len(ep['indices']) for ep in complete_episodes]
        print(f"  长度统计: 平均={np.mean(lengths):.1f}, "
              f"最小={min(lengths)}, 最大={max(lengths)}")

    return complete_episodes


def sample_trajectories(episodes, num_trajectories, seed):
    """随机采样轨迹"""
    print(f"\n随机采样 {num_trajectories} 条轨迹...")

    if len(episodes) <= num_trajectories:
        print(f"  可用轨迹不足，返回全部 {len(episodes)} 条")
        sampled = episodes
    else:
        random.seed(seed)
        sampled = random.sample(episodes, num_trajectories)
        print(f"  已采样 {len(sampled)} 条轨迹")

    return sampled


def plot_trajectory_full(trajectory_data, trajectory_id, output_dir, use_chinese=False):
    """绘制单条轨迹的完整图（动作+状态）"""
    actions = trajectory_data['actions']
    states = trajectory_data['states']
    is_human = trajectory_data['is_human']

    # 创建图形（7行5列，共35个子图位置）
    # 第1行：动作 dx, dy, dz, drx, dry
    # 第2行：动作 drz + 4个分隔区域
    # 第3-7行：19个状态
    fig, axes = plt.subplots(7, 5, figsize=(20, 16))
    fig.suptitle(f'Trajectory {trajectory_id} - Actions & States (Length: {len(actions)} steps)',
                 fontsize=16, fontweight='bold')

    steps = np.arange(len(actions))

    # 颜色
    COLOR_HUMAN = '#2E86DE'
    COLOR_ROBOT = '#EE5A24'

    legend_added = {'human': False, 'robot': False}
    legend_human = 'Human Intervention'
    legend_robot = 'Robot Execution'

    # ========== 绘制动作（前6个子图）==========
    action_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0)]

    for dim_idx in range(6):
        row, col = action_positions[dim_idx]
        ax = axes[row, col]

        action_values = actions[:, dim_idx]

        # 分段绘制
        segments = []
        start_idx = 0
        current_label = is_human[0]

        for i in range(1, len(is_human)):
            if is_human[i] != current_label:
                segments.append((start_idx, i, current_label))
                start_idx = i
                current_label = is_human[i]
        segments.append((start_idx, len(is_human), current_label))

        for start, end, is_human_segment in segments:
            color = COLOR_HUMAN if is_human_segment else COLOR_ROBOT
            label_text = legend_human if is_human_segment else legend_robot
            label_key = 'human' if is_human_segment else 'robot'

            if dim_idx == 0 and not legend_added[label_key]:
                ax.plot(steps[start:end], action_values[start:end],
                       color=color, linewidth=1.5, label=label_text)
                legend_added[label_key] = True
            else:
                ax.plot(steps[start:end], action_values[start:end],
                       color=color, linewidth=1.5)

        ax.set_title(f'ACTION: {ACTION_LABELS[dim_idx]}', fontsize=9, fontweight='bold', color='#C4612F')
        ax.set_xlabel('Step', fontsize=7)
        ax.set_ylabel('Value', fontsize=7)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.tick_params(labelsize=6)

        y_min, y_max = action_values.min(), action_values.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        else:
            ax.set_ylim(-0.1, 0.1)

    # 添加图例
    axes[0, 0].legend(loc='upper right', fontsize=7, framealpha=0.9)

    # 隐藏分隔区域
    for col in range(1, 5):
        axes[1, col].set_visible(False)

    # ========== 绘制状态（19个子图）==========
    state_positions = []
    for row in range(2, 7):
        for col in range(5):
            state_positions.append((row, col))
    state_positions = state_positions[:19]  # 只取前19个

    for dim_idx in range(19):
        row, col = state_positions[dim_idx]
        ax = axes[row, col]

        state_values = states[:, dim_idx]

        # 分段绘制
        segments = []
        start_idx = 0
        current_label = is_human[0]

        for i in range(1, len(is_human)):
            if is_human[i] != current_label:
                segments.append((start_idx, i, current_label))
                start_idx = i
                current_label = is_human[i]
        segments.append((start_idx, len(is_human), current_label))

        for start, end, is_human_segment in segments:
            color = COLOR_HUMAN if is_human_segment else COLOR_ROBOT
            ax.plot(steps[start:end], state_values[start:end],
                   color=color, linewidth=1.5)

        ax.set_title(f'STATE: {STATE_NAMES[dim_idx]}', fontsize=8, fontweight='bold', color='#2E86DE')
        ax.set_xlabel('Step', fontsize=6)
        ax.set_ylabel('Value', fontsize=6)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=6)

        y_min, y_max = state_values.min(), state_values.max()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        else:
            ax.set_ylim(y_min - 0.1, y_max + 0.1)

    # 隐藏最后一个空位置
    axes[6, 4].set_visible(False)

    # 调整布局
    plt.tight_layout()

    # 保存
    output_path = Path(output_dir) / f'trajectory_{trajectory_id}_full.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_path}")


def main():
    args = parse_args()

    print("="*80)
    print("Demo 轨迹完整可视化工具（动作 + 状态）")
    print("="*80)

    # 确定文件路径
    if args.buffer_file and args.demo_file:
        buffer_file = args.buffer_file
        demo_file = args.demo_file
    else:
        buffer_file, demo_file = load_latest_files(args.checkpoint_dir)

    # 加载数据
    transitions, is_human = load_and_mark_human(buffer_file, demo_file)

    # 切分 episodes
    episodes = split_episodes(transitions, is_human)

    if not episodes:
        print("\n错误: 没有找到完整的 episode")
        return 1

    # 采样轨迹
    sampled_episodes = sample_trajectories(episodes, args.num_trajectories, args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 绘制每条轨迹
    print("\n开始绘图...")
    for traj_id, episode in enumerate(sampled_episodes):
        indices = episode['indices']

        # 提取动作和状态
        actions = np.array([transitions[i]['actions'][:6] for i in indices])
        states = np.array([transitions[i]['observations']['state'].flatten()
                          for i in indices])
        is_human_traj = np.array(episode['is_human'])

        # 准备数据
        trajectory_data = {
            'actions': actions,
            'states': states,
            'is_human': is_human_traj,
        }

        # 绘图
        plot_trajectory_full(trajectory_data, traj_id, output_dir, args.use_chinese)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n已生成 {len(sampled_episodes)} 个完整轨迹图表")
    print(f"保存位置: {output_dir}")
    print("\n提示:")
    print("  - 蓝色线段：人工干预")
    print("  - 橙色线段：机器人执行")
    print("  - 每个图包含 6 个动作 + 19 个状态子图")

    return 0


if __name__ == '__main__':
    sys.exit(main())
