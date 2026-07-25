#!/usr/bin/env python3
"""
Buffer 数据检查工具 - Inspect Buffer Data

功能：
    快速检查 replay buffer 的内容，显示数据统计和样本。
    用于调试、验证数据格式和理解 buffer 结构。

主要功能：
    1. 显示 buffer 文件列表和大小
    2. 加载并分析单个文件
    3. 显示 transition 结构
    4. 统计 episode 信息
    5. 可选：保存样本图像

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 检查最新的 buffer 文件
    python utils/inspect_buffer.py

    # 检查指定文件
    python utils/inspect_buffer.py --file checkpoints/buffer/transitions_1000.pkl

    # 保存前 10 帧的图像
    python utils/inspect_buffer.py --save-images --num-images 10

    # 显示详细的 episode 统计
    python utils/inspect_buffer.py --show-episodes
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def format_bytes(bytes_size):
    """格式化字节大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查 HIL-SERL replay buffer 数据格式和内容"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="examples/experiments/marvin_usb_insertion/checkpoints",
        help="Checkpoints 目录路径"
    )
    parser.add_argument(
        "--file",
        help="指定要检查的 buffer 文件路径（默认：最新文件）"
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="保存样本图像到 utils/buffer_samples/ 目录"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="保存的图像数量（默认：5）"
    )
    parser.add_argument(
        "--show-episodes",
        action="store_true",
        help="显示 episode 切分和统计"
    )
    return parser.parse_args()


def list_buffer_files(checkpoint_dir):
    """列出所有 buffer 文件"""
    buffer_dir = os.path.join(checkpoint_dir, "buffer")
    files = sorted(
        glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")),
        key=lambda x: int(x.split('_')[-1].replace('.pkl', ''))
    )
    return files


def load_transitions(file_path):
    """加载 transitions"""
    print(f"\n加载文件: {file_path}")
    with open(file_path, 'rb') as f:
        transitions = pickle.load(f)
    file_size = os.path.getsize(file_path)
    print(f"文件大小: {format_bytes(file_size)}")
    print(f"Transition 数量: {len(transitions)}")
    return transitions


def analyze_structure(transitions):
    """分析数据结构"""
    if not transitions:
        print("警告: 文件为空")
        return

    sample = transitions[0]

    print("\n" + "="*80)
    print("单个 Transition 的结构")
    print("="*80)

    print(f"\nTransition 类型: {type(sample)}")
    print(f"顶层键: {list(sample.keys())}")

    # Observations
    print("\n" + "-"*80)
    print("observations:")
    print("-"*80)
    obs = sample['observations']
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={str(value.shape):25s} dtype={value.dtype}")
        else:
            print(f"  {key:20s}: type={type(value)}")

    # Actions
    print("\n" + "-"*80)
    print("actions:")
    print("-"*80)
    actions = sample['actions']
    print(f"  shape: {actions.shape}")
    print(f"  dtype: {actions.dtype}")
    print(f"  范围: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  值: {actions}")

    # Next observations
    print("\n" + "-"*80)
    print("next_observations:")
    print("-"*80)
    next_obs = sample['next_observations']
    for key, value in next_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={str(value.shape):25s} dtype={value.dtype}")

    # Rewards, dones, etc.
    print("\n" + "-"*80)
    print("其他字段:")
    print("-"*80)
    for key in ['rewards', 'dones', 'masks', 'grasp_penalty']:
        if key in sample:
            value = sample[key]
            print(f"  {key:20s}: {value} (type: {type(value).__name__})")


def analyze_statistics(transitions):
    """统计分析"""
    print("\n" + "="*80)
    print("数据统计")
    print("="*80)

    # 提取数据
    actions = np.array([t['actions'] for t in transitions])
    rewards = np.array([t['rewards'] for t in transitions])
    dones = np.array([t['dones'] for t in transitions])
    states = np.array([t['observations']['state'].flatten() for t in transitions])

    # 动作统计
    print("\n动作统计 (7维):")
    action_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']
    print(f"{'维度':<10}{'最小值':>10}{'最大值':>10}{'平均值':>10}{'标准差':>10}")
    for i, name in enumerate(action_names):
        vals = actions[:, i]
        print(f"{name:<10}{vals.min():>10.3f}{vals.max():>10.3f}"
              f"{vals.mean():>10.3f}{vals.std():>10.3f}")

    # 奖励统计
    print(f"\n奖励统计:")
    print(f"  总步数: {len(rewards)}")
    print(f"  奖励=1 的步数: {(rewards > 0).sum()}")
    print(f"  奖励=0 的步数: {(rewards == 0).sum()}")
    print(f"  平均奖励: {rewards.mean():.4f}")

    # Episode 统计
    print(f"\nEpisode 统计:")
    num_dones = dones.sum()
    print(f"  done=1 的步数: {num_dones}")
    if num_dones > 0:
        print(f"  完整 episode 数量: {num_dones}")

    # State 统计
    print(f"\nState 统计 ({states.shape[1]} 维):")
    if states.shape[1] == 19:
        state_names = [
            ('gripper', 0, 1),
            ('force', 1, 4),
            ('tcp_pose', 4, 10),
            ('torque', 10, 13),
            ('tcp_vel', 13, 19),
        ]
    elif states.shape[1] == 13:
        state_names = [
            ('gripper', 0, 1),
            ('tcp_pose', 1, 7),
            ('tcp_vel', 7, 13),
        ]
    else:
        state_names = []

    if state_names:
        print(f"{'分量':<15}{'维度':>8}{'最小值':>12}{'最大值':>12}{'平均值':>12}")
        for name, start, end in state_names:
            vals = states[:, start:end]
            print(f"{name:<15}{end-start:>8}{vals.min():>12.4f}"
                  f"{vals.max():>12.4f}{vals.mean():>12.4f}")


def save_sample_images(transitions, num_images, output_dir):
    """保存样本图像"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n保存前 {num_images} 个样本的图像到: {output_dir}")

    for i in range(min(num_images, len(transitions))):
        t = transitions[i]
        obs = t['observations']

        # 为每个相机保存图像
        for cam_name in ['wrist_1', 'wrist_2', 'side_policy', 'side_classifier']:
            if cam_name not in obs:
                continue

            img = obs[cam_name][0]  # 去掉 batch 维度

            # RGB to BGR (OpenCV 格式)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 添加文本标注
            text = f"Frame {i} | {cam_name}"
            cv2.putText(img_bgr, text, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # 保存
            filename = output_dir / f"frame_{i:03d}_{cam_name}.jpg"
            cv2.imwrite(str(filename), img_bgr)

    print(f"✓ 已保存 {min(num_images, len(transitions))} 个样本的图像")


def analyze_episodes(transitions):
    """分析 episode 切分"""
    print("\n" + "="*80)
    print("Episode 分析")
    print("="*80)

    episodes = []
    current_episode = []

    for i, t in enumerate(transitions):
        current_episode.append(i)
        if t['dones'] == 1:
            episodes.append(current_episode)
            current_episode = []

    # 未完成的 episode
    if current_episode:
        episodes.append(current_episode)

    print(f"\n完整 episode 数量: {len(episodes)-1 if current_episode else len(episodes)}")
    if current_episode:
        print(f"未完成 episode: 1 (长度 {len(current_episode)})")

    # Episode 长度统计
    complete_episodes = episodes[:-1] if current_episode else episodes
    if complete_episodes:
        lengths = [len(ep) for ep in complete_episodes]
        print(f"\nEpisode 长度统计:")
        print(f"  平均: {np.mean(lengths):.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")
        print(f"  最小: {min(lengths)}")
        print(f"  最大: {max(lengths)}")

        # 显示前几个 episode
        print(f"\n前 {min(5, len(complete_episodes))} 个 episode:")
        for i, ep in enumerate(complete_episodes[:5]):
            # 检查是否有奖励
            has_reward = any(transitions[idx]['rewards'] > 0 for idx in ep)
            success_str = "✓ 成功" if has_reward else "✗ 失败"
            print(f"  Episode {i+1}: 长度={len(ep):3d}, {success_str}")


def main():
    args = parse_args()

    print("="*80)
    print("HIL-SERL Replay Buffer 检查工具")
    print("="*80)

    # 确定要检查的文件
    if args.file:
        file_path = args.file
    else:
        # 使用最新的文件
        files = list_buffer_files(args.checkpoint_dir)
        if not files:
            print(f"错误: 没有找到 buffer 文件在 {args.checkpoint_dir}/buffer/")
            return 1
        file_path = files[-1]
        print(f"\n找到 {len(files)} 个 buffer 文件")
        print(f"使用最新文件: {os.path.basename(file_path)}")

    # 加载数据
    try:
        transitions = load_transitions(file_path)
    except Exception as e:
        print(f"错误: 无法加载文件: {e}")
        return 1

    # 分析结构
    analyze_structure(transitions)

    # 统计分析
    analyze_statistics(transitions)

    # Episode 分析
    if args.show_episodes:
        analyze_episodes(transitions)

    # 保存图像
    if args.save_images:
        output_dir = "utils/buffer_samples"
        save_sample_images(transitions, args.num_images, output_dir)

    print("\n" + "="*80)
    print("检查完成！")
    print("="*80)
    print("\n提示:")
    print("  - 详细格式说明请查看: utils/BUFFER_FORMAT.md")
    print("  - 分析动作抖动请使用: python utils/analyze_policy_jitter.py")
    print("  - 保存图像请添加: --save-images")
    print("  - 显示 episode 请添加: --show-episodes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
