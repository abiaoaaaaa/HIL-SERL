#!/usr/bin/env python3
"""
可视化评估动作轨迹。

每个 episode 一张图，不同动作维度各占一个子图。
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = ("dx", "dy", "dz", "drz", "gripper")

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["AR PL UKai CN", "AR PL UMing CN", "WenQuanYi Micro Hei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_JSON_PATH = (
    "/home/xlb/code_marvin/hil-serl/examples/experiments/"
    "marvin_usb_insertion/checkpoints/eval_actions_124000.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description="可视化评估中的动作轨迹")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help="动作 JSON 文件路径 (eval_actions_*.json)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="只显示指定episode编号（默认：显示全部）",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="最多显示N个episode，0=全部（默认：0）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存为PNG而不是显示",
    )
    return parser.parse_args()


def plot_one_episode(ep: dict) -> plt.Figure:
    """为一个 episode 创建一张图，5 个动作各占一个子图。"""
    actions = np.array(ep["actions"])  # [T, 5]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), squeeze=False)
    axes_flat = axes.flatten()

    for j in range(5):
        ax = axes_flat[j]
        ax.plot(actions[:, j], linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(ACTION_NAMES[j])
        ax.set_xlabel("步数")
        ax.set_ylabel("动作值")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    # 隐藏第6个空白子图
    axes_flat[5].set_visible(False)

    fig.suptitle(
        f"Ep{ep['episode']} {'✓' if ep['success'] else '✗'} ({ep['steps']}步)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_action_trajectory(
    json_path: str,
    episode: int | None,
    max_episodes: int,
    save: bool,
):
    with open(json_path, "r") as f:
        data = json.load(f)

    episodes = data["episodes"]

    # 筛选
    if episode is not None:
        episodes = [ep for ep in episodes if ep["episode"] == episode]
        if not episodes:
            print(f"错误: 未找到 episode={episode}")
            return 1
    elif max_episodes > 0:
        episodes = episodes[:max_episodes]

    base_name = os.path.splitext(json_path)[0]  # 去掉 .json

    for ep in episodes:
        fig = plot_one_episode(ep)

        if save:
            output_path = f"{base_name}_ep{ep['episode']}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"已保存到: {output_path}")
        else:
            plt.show()

    print(
        f"共 {len(episodes)} 个 episode，"
        f"checkpoint={data['checkpoint_step']}，"
        f"成功率={data['success_rate']:.1%}"
    )
    return 0


def main():
    args = parse_args()
    if not os.path.exists(args.json_path):
        print(f"错误: 文件不存在: {args.json_path}")
        return 1
    return plot_action_trajectory(
        args.json_path, args.episode, args.max_episodes, args.save
    )


if __name__ == "__main__":
    raise SystemExit(main())
