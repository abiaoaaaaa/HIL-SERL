#!/usr/bin/env python3
"""
分析 HIL-SERL replay buffer 中的 episode 和动作抖动。

默认分析当前 Marvin USB 训练的全部已落盘分片。程序逐个加载 pickle，
因此不会同时把所有相机图像放进内存。

示例：
    python utility/analyze_policy_jitter.py
    python utility/analyze_policy_jitter.py --last-n-files 5
    python utility/analyze_policy_jitter.py --checkpoint-dir /path/to/checkpoints

注意：
    pickle 只能用于可信数据。本脚本假设输入是本机训练程序生成的 buffer。
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import os
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


DEFAULT_CHECKPOINT_DIR = (
    "/home/xlb/code_marvin/hil-serl/examples/experiments/"
    "marvin_usb_insertion/checkpoints"
)
ACTION_NAMES_7D = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
ACTION_NAMES_5D = ("dx", "dy", "dz", "drz", "gripper")

# 根据数据自动选择
ACTION_NAMES = ACTION_NAMES_5D  # 默认5D，run()中会动态覆盖


@dataclass
class LoadedData:
    actions: np.ndarray
    states: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
    human: np.ndarray
    selected_files: list[str]
    unmatched_demo: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 episode 数量、Policy动作抖动和真机末端位移抖动。"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_CHECKPOINT_DIR,
        help="包含 buffer/ 和 demo_buffer/ 的 checkpoints 目录。",
    )
    parser.add_argument(
        "--last-n-files",
        type=int,
        default=0,
        help="只分析最后N个分片；0表示分析全部分片（默认：0）。",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.03,
        help="计算动作方向翻转时忽略的小动作阈值（默认：0.03）。",
    )
    parser.add_argument(
        "--top-episodes",
        type=int,
        default=10,
        help="显示Policy最抖的前N个完整episode（默认：10）。",
    )
    return parser.parse_args()


def step_from_path(path: str) -> int:
    match = re.search(r"(\d+)\.pkl$", path)
    if match is None:
        raise ValueError(f"无法从文件名提取step: {path}")
    return int(match.group(1))


def flat_state(observation: dict) -> np.ndarray:
    return np.asarray(observation["state"]).reshape(-1)


def transition_signature(transition: dict) -> bytes:
    """匹配主buffer和demo_buffer中的同一条transition，不读取图像内容。"""
    digest = hashlib.blake2b(digest_size=16)
    values = (
        flat_state(transition["observations"]),
        np.asarray(transition["actions"]).reshape(-1),
        flat_state(transition["next_observations"]),
    )
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.digest()


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def select_file_pairs(
    checkpoint_dir: str, last_n_files: int
) -> list[tuple[str, str]]:
    buffer_dir = os.path.join(checkpoint_dir, "buffer")
    demo_dir = os.path.join(checkpoint_dir, "demo_buffer")
    files = sorted(
        glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")),
        key=step_from_path,
    )
    if last_n_files > 0:
        files = files[-last_n_files:]

    pairs = []
    for buffer_path in files:
        demo_path = os.path.join(
            demo_dir, os.path.basename(buffer_path)
        )
        if not os.path.exists(demo_path):
            print(
                f"[跳过] 缺少对应demo_buffer: {demo_path}",
                file=sys.stderr,
            )
            continue
        pairs.append((buffer_path, demo_path))
    return pairs


def load_data(file_pairs: Iterable[tuple[str, str]]) -> LoadedData:
    all_actions = []
    all_states = []
    all_next_states = []
    all_dones = []
    all_rewards = []
    all_human = []
    selected_files = []
    unmatched_demo = 0

    for buffer_path, demo_path in file_pairs:
        step = step_from_path(buffer_path)
        try:
            transitions = load_pickle(buffer_path)
        except Exception as exc:
            print(
                f"[跳过] 主buffer读取失败，可能仍在写入: "
                f"{buffer_path}: {exc}",
                file=sys.stderr,
            )
            continue

        try:
            actions = np.stack(
                [
                    np.asarray(item["actions"]).reshape(-1)
                    for item in transitions
                ]
            )
            states = np.stack(
                [flat_state(item["observations"]) for item in transitions]
            )
            next_states = np.stack(
                [
                    flat_state(item["next_observations"])
                    for item in transitions
                ]
            )
            dones = np.asarray(
                [bool(item["dones"]) for item in transitions],
                dtype=bool,
            )
            rewards = np.asarray(
                [float(item["rewards"]) for item in transitions],
                dtype=np.float64,
            )
            signatures = [
                transition_signature(item) for item in transitions
            ]
        except Exception as exc:
            print(
                f"[跳过] 主buffer结构无法解析: {buffer_path}: {exc}",
                file=sys.stderr,
            )
            del transitions
            gc.collect()
            continue

        transition_count = len(transitions)
        del transitions
        gc.collect()

        try:
            demo_transitions = load_pickle(demo_path)
            demo_counts = Counter(
                transition_signature(item) for item in demo_transitions
            )
            demo_count = len(demo_transitions)
        except Exception as exc:
            print(
                f"[跳过] demo_buffer读取失败，无法区分人工与Policy: "
                f"{demo_path}: {exc}",
                file=sys.stderr,
            )
            del actions, states, next_states, dones, rewards, signatures
            gc.collect()
            continue

        del demo_transitions
        gc.collect()

        human = np.zeros(transition_count, dtype=bool)
        for index, signature in enumerate(signatures):
            if demo_counts[signature] > 0:
                human[index] = True
                demo_counts[signature] -= 1

        missing = sum(demo_counts.values())
        unmatched_demo += missing
        print(
            f"[读取] step={step:<7d} "
            f"buffer={transition_count:<5d} "
            f"人工={int(human.sum()):<5d} "
            f"demo未匹配={missing}"
        )

        all_actions.append(actions)
        all_states.append(states)
        all_next_states.append(next_states)
        all_dones.append(dones)
        all_rewards.append(rewards)
        all_human.append(human)
        selected_files.append(buffer_path)

    if not all_actions:
        raise RuntimeError("没有成功读取任何成对的buffer/demo_buffer分片。")

    return LoadedData(
        actions=np.concatenate(all_actions),
        states=np.concatenate(all_states),
        next_states=np.concatenate(all_next_states),
        dones=np.concatenate(all_dones),
        rewards=np.concatenate(all_rewards),
        human=np.concatenate(all_human),
        selected_files=selected_files,
        unmatched_demo=unmatched_demo,
    )


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def percent(mask: np.ndarray) -> float:
    if mask.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(mask))


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def print_title(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def infer_state_layout(state_dim: int) -> tuple[Optional[slice], Optional[slice]]:
    """
    SERLObsWrapper使用gym Dict的字母序flatten：

    旧版 (无 last_action):
      19D: gripper_pose(1), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6)
      13D: gripper_pose(1), tcp_pose(6), tcp_vel(6)

    新版 (含 last_action):
      24D: gripper_pose(1), last_action(5), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6)
    """
    if state_dim == 24:
        # 新版：gripper(1) + last_action(5) + force(3) + pose(6) + torque(3) + vel(6)
        return slice(9, 15), slice(18, 24)
    if state_dim == 19:
        # 旧版：gripper(1) + force(3) + pose(6) + torque(3) + vel(6)
        return slice(4, 10), slice(13, 19)
    if state_dim == 13:
        return slice(1, 7), slice(7, 13)
    return None, None


def continuity_mask(data: LoadedData) -> np.ndarray:
    continuous = np.zeros(len(data.actions), dtype=bool)
    if len(continuous) <= 1:
        return continuous
    states_match = np.all(
        np.isclose(
            data.next_states[:-1],
            data.states[1:],
            atol=1e-6,
            rtol=1e-5,
        ),
        axis=1,
    )
    continuous[1:] = (~data.dones[:-1]) & states_match
    return continuous


def split_episode_segments(
    dones: np.ndarray, continuous: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], Optional[tuple[int, int]]]:
    """
    同时使用done和观测连续性切分episode。

    仅按done切分会把进程重启前后的两段轨迹拼在一起，因为actor被中断时
    最后一条transition未必带done=True。
    """
    completed = []
    interrupted = []
    start = 0
    for index in range(len(dones)):
        next_is_discontinuous = (
            index + 1 < len(dones) and not continuous[index + 1]
        )
        if dones[index]:
            completed.append((start, index + 1))
            start = index + 1
        elif next_is_discontinuous:
            interrupted.append((start, index + 1))
            start = index + 1

    trailing = (start, len(dones)) if start < len(dones) else None
    return completed, interrupted, trailing


def print_dataset_summary(
    data: LoadedData,
    continuous: np.ndarray,
    full_history: bool,
) -> list[tuple[int, int]]:
    print_title("1. 数据和Episode概况")
    ranges, interrupted, trailing = split_episode_segments(
        data.dones, continuous
    )
    length_ranges = ranges
    if not full_history and length_ranges and length_ranges[0][0] == 0:
        # 最近N个分片通常从某个episode中间开始，第一段长度是左截断的。
        length_ranges = length_ranges[1:]
    episode_lengths = np.asarray(
        [end - start for start, end in length_ranges], dtype=np.int64
    )
    successes = sum(
        bool(np.any(data.rewards[start:end] > 0))
        for start, end in ranges
    )

    print(
        f"分片数量: {len(data.selected_files)}  "
        f"范围: {os.path.basename(data.selected_files[0])} -> "
        f"{os.path.basename(data.selected_files[-1])}"
    )
    print(
        f"Transition总数: {len(data.actions)}  "
        f"Action shape: {tuple(data.actions.shape)}  "
        f"State shape: {tuple(data.states.shape)}"
    )
    print(
        f"窗口内正常结束的Episode数(dones=True): {len(ranges)}  "
        f"成功Episode: {successes}  "
        f"成功率: {fmt(100.0 * successes / len(ranges), 1) if ranges else 'N/A'}%"
    )
    if episode_lengths.size:
        print(
            "Episode长度(步) "
            f"平均/中位数/P95/最小/最大: "
            f"{fmt(mean(episode_lengths), 1)}/"
            f"{fmt(median(episode_lengths), 1)}/"
            f"{fmt(percentile(episode_lengths, 95), 1)}/"
            f"{int(episode_lengths.min())}/"
            f"{int(episode_lengths.max())}"
        )
    interrupted_steps = sum(end - start for start, end in interrupted)
    trailing_steps = 0 if trailing is None else trailing[1] - trailing[0]
    print(
        f"因进程重启/数据不连续而中断的片段: "
        f"{len(interrupted)}段，共{interrupted_steps}步"
    )
    print(f"末尾尚未完成的Episode片段: {trailing_steps}步")
    print(
        f"Policy步数: {int((~data.human).sum())}  "
        f"人工干预步数: {int(data.human.sum())}  "
        f"人工比例: {fmt(percent(data.human), 1)}%"
    )
    if data.unmatched_demo:
        print(f"警告: 有{data.unmatched_demo}条demo transition未匹配。")
    return ranges


def print_action_size(data: LoadedData) -> None:
    print_title("2. Policy与人工动作大小")
    is_7d = data.actions.shape[1] == 7
    rot_cols = slice(3, 6) if is_7d else 3  # 7D: 3轴旋转, 5D: drz
    rot_label = "旋转Norm" if is_7d else "旋转(drz)"
    act_cols = slice(0, 6) if is_7d else slice(0, 4)  # 不含gripper

    print(
        f"{'来源':<10}{'步数':>8}{'平移Norm中位':>16}"
        f"{'平移Norm P95':>15}{rot_label + '中位':>16}"
        f"{rot_label + 'P95':>15}{'任一维饱和':>14}"
    )
    for name, mask in (("Policy", ~data.human), ("人工", data.human)):
        actions = data.actions[mask]
        trans_norm = np.linalg.norm(actions[:, :3], axis=1)
        if is_7d:
            rot_abs = np.linalg.norm(actions[:, 3:6], axis=1)
        else:
            rot_abs = np.abs(actions[:, 3])
        saturation = np.any(np.abs(actions[:, act_cols]) > 0.95, axis=1)
        print(
            f"{name:<10}{len(actions):>8d}"
            f"{fmt(median(trans_norm)):>16}"
            f"{fmt(percentile(trans_norm, 95)):>15}"
            f"{fmt(median(rot_abs)):>16}"
            f"{fmt(percentile(rot_abs, 95)):>15}"
            f"{(fmt(percent(saturation), 1) + '%'):>14}"
        )

    print()
    print("Policy各动作维统计：")
    policy_actions = data.actions[~data.human]
    print(
        f"{'维度':<10}{'平均':>10}{'标准差':>10}"
        f"{'P05':>10}{'中位数':>10}{'P95':>10}"
    )
    for index, name in enumerate(ACTION_NAMES):
        values = policy_actions[:, index]
        print(
            f"{name:<10}{fmt(mean(values)):>10}"
            f"{fmt(float(np.std(values))):>10}"
            f"{fmt(percentile(values, 5)):>10}"
            f"{fmt(median(values)):>10}"
            f"{fmt(percentile(values, 95)):>10}"
        )


def vector_reversal_rate(
    previous: np.ndarray, current: np.ndarray, deadband: float
) -> float:
    previous_norm = np.linalg.norm(previous, axis=1)
    current_norm = np.linalg.norm(current, axis=1)
    valid = (previous_norm > deadband) & (current_norm > deadband)
    if not np.any(valid):
        return float("nan")
    reversed_direction = np.sum(previous * current, axis=1) < 0
    return percent(reversed_direction[valid])


def pair_categories(data: LoadedData, continuous: np.ndarray):
    indices = np.flatnonzero(continuous)
    previous_human = data.human[indices - 1]
    current_human = data.human[indices]
    return indices, {
        "Policy→Policy": (~previous_human) & (~current_human),
        "人工→人工": previous_human & current_human,
        "Policy→人工": (~previous_human) & current_human,
        "人工→Policy": previous_human & (~current_human),
    }


def print_action_jitter(
    data: LoadedData, continuous: np.ndarray, deadband: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    print_title("3. 相邻动作抖动（核心指标）")
    indices, categories = pair_categories(data, continuous)
    is_7d = data.actions.shape[1] == 7
    act_dim = 6 if is_7d else 4  # 7D: dx,dy,dz,drx,dry,drz; 5D: dx,dy,dz,drz
    dim_label = "6D" if is_7d else "4D"

    print(
        f"{'相邻来源':<16}{'对数':>8}{dim_label + '跳变均值':>14}"
        f"{dim_label + '跳变P95':>14}{'平移跳变均值':>16}"
        f"{'平移跳变P95':>15}{'平移反向':>12}"
    )
    for name, category_mask in categories.items():
        selected = indices[category_mask]
        previous = data.actions[selected - 1]
        current = data.actions[selected]
        jump_act = np.linalg.norm(
            current[:, :act_dim] - previous[:, :act_dim], axis=1
        )
        jump_translation = np.linalg.norm(
            current[:, :3] - previous[:, :3], axis=1
        )
        reversal = vector_reversal_rate(
            previous[:, :3], current[:, :3], deadband
        )
        print(
            f"{name:<16}{len(selected):>8d}"
            f"{fmt(mean(jump_act)):>14}"
            f"{fmt(percentile(jump_act, 95)):>14}"
            f"{fmt(mean(jump_translation)):>16}"
            f"{fmt(percentile(jump_translation, 95)):>15}"
            f"{(fmt(reversal, 1) + '%'):>12}"
        )

    policy_pairs = indices[categories["Policy→Policy"]]
    print()
    print(
        f"Policy连续动作各轴符号翻转率"
        f"（两步绝对值都>{deadband}才计入）："
    )
    is_7d = data.actions.shape[1] == 7
    n_non_gripper = 6 if is_7d else 4  # 7D: dx,dy,dz,drx,dry,drz; 5D: dx,dy,dz,drz
    for axis, name in enumerate(ACTION_NAMES[:n_non_gripper]):
        previous = data.actions[policy_pairs - 1, axis]
        current = data.actions[policy_pairs, axis]
        valid = (
            (np.abs(previous) > deadband)
            & (np.abs(current) > deadband)
        )
        rate = (
            percent((previous[valid] * current[valid]) < 0)
            if np.any(valid)
            else float("nan")
        )
        print(f"  {name:<4}: {fmt(rate, 1)}%")

    source_switches = categories["Policy→人工"] | categories["人工→Policy"]
    print(
        f"\n控制来源切换: {int(source_switches.sum())}/"
        f"{len(indices)} 对 ({fmt(percent(source_switches), 1)}%)"
    )
    return indices, categories


def print_observed_motion(
    data: LoadedData,
    indices: np.ndarray,
    categories: dict[str, np.ndarray],
) -> None:
    pose_slice, velocity_slice = infer_state_layout(data.states.shape[1])
    print_title("4. 真机末端实际位移抖动")
    if pose_slice is None:
        print(
            f"State维数为{data.states.shape[1]}，无法自动确定tcp_pose索引，"
            "跳过实际位移分析。"
        )
        return

    displacement_mm = (
        data.next_states[:, pose_slice.start : pose_slice.start + 3]
        - data.states[:, pose_slice.start : pose_slice.start + 3]
    ) * 1000.0

    print(
        f"{'相邻来源':<16}{'对数':>8}{'位移跳变均值mm':>18}"
        f"{'位移跳变P95mm':>18}{'实际位移反向':>16}"
    )
    for name, category_mask in categories.items():
        selected = indices[category_mask]
        previous = displacement_mm[selected - 1]
        current = displacement_mm[selected]
        jumps = np.linalg.norm(current - previous, axis=1)
        reversal = vector_reversal_rate(previous, current, deadband=0.2)
        print(
            f"{name:<16}{len(selected):>8d}"
            f"{fmt(mean(jumps), 2):>18}"
            f"{fmt(percentile(jumps, 95), 2):>18}"
            f"{(fmt(reversal, 1) + '%'):>16}"
        )

    print()
    for name, mask in (("Policy", ~data.human), ("人工", data.human)):
        magnitude = np.linalg.norm(displacement_mm[mask], axis=1)
        print(
            f"{name}每步实际移动距离 中位数/P95/最大值: "
            f"{fmt(median(magnitude), 2)}/"
            f"{fmt(percentile(magnitude, 95), 2)}/"
            f"{fmt(float(np.max(magnitude)), 2)} mm"
        )

    if velocity_slice is not None:
        print(
            "说明: 实际位移来自相邻obs的tcp_pose，"
            "没有使用相机图像，也不依赖动作坐标系。"
        )


def print_gripper_summary(data: LoadedData) -> None:
    print_title("5. 夹爪命令与潜在阻塞")
    gripper_idx = 6 if data.actions.shape[1] == 7 else 4
    gripper_action = data.actions[:, gripper_idx]
    gripper_position = data.states[:, 0]  # 字母序: gripper_pose 在索引0
    next_gripper_position = data.next_states[:, 0]
    for name, mask in (("Policy", ~data.human), ("人工", data.human)):
        strong = (np.abs(gripper_action) > 0.5) & mask
        wants_close = (
            (gripper_action <= -0.5)
            & (gripper_position > 0.5)
            & mask
        )
        wants_open = (
            (gripper_action >= 0.5)
            & (gripper_position < 0.5)
            & mask
        )
        state_changed = (
            (np.abs(next_gripper_position - gripper_position) > 0.2)
            & mask
        )
        source_count = int(mask.sum())
        print(
            f"{name}: 强命令={int(strong.sum())}/{source_count} "
            f"({fmt(100.0 * strong.sum() / source_count, 1)}%), "
            f"满足开关条件={int((wants_close | wants_open).sum())}, "
            f"观测到夹爪变化={int(state_changed.sum())}"
        )


def episode_rows(
    data: LoadedData,
    ranges: list[tuple[int, int]],
    continuous: np.ndarray,
    deadband: float,
):
    rows = []
    for episode_id, (start, end) in enumerate(ranges, start=1):
        local_indices = np.arange(max(start + 1, 1), end)
        policy_pairs = local_indices[
            continuous[local_indices]
            & (~data.human[local_indices - 1])
            & (~data.human[local_indices])
        ]
        if policy_pairs.size:
            previous = data.actions[policy_pairs - 1, :3]
            current = data.actions[policy_pairs, :3]
            jumps = np.linalg.norm(current - previous, axis=1)
            reversal = vector_reversal_rate(
                previous, current, deadband
            )
            jump_mean = mean(jumps)
            jump_p95 = percentile(jumps, 95)
        else:
            jump_mean = float("nan")
            jump_p95 = float("nan")
            reversal = float("nan")
        rows.append(
            {
                "episode": episode_id,
                "length": end - start,
                "success": bool(np.any(data.rewards[start:end] > 0)),
                "policy_steps": int((~data.human[start:end]).sum()),
                "human_percent": percent(data.human[start:end]),
                "policy_pairs": int(policy_pairs.size),
                "jump_mean": jump_mean,
                "jump_p95": jump_p95,
                "reversal": reversal,
            }
        )
    return rows


def print_jitter_trend_by_epochs(
    data: LoadedData,
    ranges: list[tuple[int, int]],
    continuous: np.ndarray,
    deadband: float,
    epoch_size: int = 100,
) -> None:
    """统计每N个episode的抖动趋势，判断是否随训练降低"""
    print_title("6. 每100个Episode的跳变趋势（训练改善分析）")

    rows = episode_rows(data, ranges, continuous, deadband)
    valid_rows = [
        row for row in rows
        if row["policy_pairs"] >= 5 and np.isfinite(row["jump_mean"])
    ]

    if len(valid_rows) < epoch_size:
        print(f"有效Episode数({len(valid_rows)})不足{epoch_size}，无法分epoch统计。")
        return

    # 按epoch_size分组
    epochs = []
    for i in range(0, len(valid_rows), epoch_size):
        epoch_rows = valid_rows[i:i+epoch_size]
        if len(epoch_rows) < 10:  # 最后一组太少则跳过
            continue

        epoch_num = i // epoch_size + 1
        jumps = [row["jump_mean"] for row in epoch_rows]
        reversals = [row["reversal"] for row in epoch_rows if np.isfinite(row["reversal"])]
        successes = [row["success"] for row in epoch_rows]

        epochs.append({
            "epoch": epoch_num,
            "episode_range": f"{epoch_rows[0]['episode']}-{epoch_rows[-1]['episode']}",
            "count": len(epoch_rows),
            "jump_mean": mean(np.array(jumps)),
            "jump_median": median(np.array(jumps)),
            "jump_p95": percentile(np.array(jumps), 95),
            "reversal_mean": mean(np.array(reversals)) if reversals else float("nan"),
            "success_rate": 100.0 * sum(successes) / len(successes),
        })

    if not epochs:
        print("无法生成epoch统计。")
        return

    print(f"\n每{epoch_size}个Episode的统计（只计入Policy对≥5的Episode）：")
    print(
        f"{'Epoch':>7}{'Episode范围':>20}{'样本数':>10}"
        f"{'跳变均值':>12}{'跳变中位':>12}{'跳变P95':>12}"
        f"{'反向率%':>12}{'成功率%':>12}"
    )

    for ep in epochs:
        print(
            f"{ep['epoch']:>7d}{ep['episode_range']:>20}{ep['count']:>10d}"
            f"{fmt(ep['jump_mean']):>12}{fmt(ep['jump_median']):>12}"
            f"{fmt(ep['jump_p95']):>12}{fmt(ep['reversal_mean'], 1):>12}"
            f"{fmt(ep['success_rate'], 1):>12}"
        )

    # 趋势分析
    print()
    if len(epochs) >= 3:
        first_third = epochs[:len(epochs)//3]
        last_third = epochs[-len(epochs)//3:]

        first_jump = mean(np.array([e["jump_mean"] for e in first_third]))
        last_jump = mean(np.array([e["jump_mean"] for e in last_third]))
        first_success = mean(np.array([e["success_rate"] for e in first_third]))
        last_success = mean(np.array([e["success_rate"] for e in last_third]))

        improvement = (first_jump - last_jump) / first_jump * 100 if first_jump > 0 else 0

        print(f"趋势分析（前1/3 vs 后1/3）：")
        print(f"  前期平均跳变: {fmt(first_jump)}  →  后期平均跳变: {fmt(last_jump)}")
        print(f"  改善幅度: {fmt(improvement, 1)}%")
        print(f"  前期成功率: {fmt(first_success, 1)}%  →  后期成功率: {fmt(last_success, 1)}%")

        if improvement > 20:
            print("  ✅ 判断: 跳变显著降低，Policy在学习平滑控制")
        elif improvement > 10:
            print("  ○ 判断: 跳变有所降低，但改善不明显")
        elif improvement > -10:
            print("  ⚠️ 判断: 跳变基本持平，Policy未学到平滑性")
        else:
            print("  ❌ 判断: 跳变反而增加，可能训练不稳定")
    else:
        print(f"只有{len(epochs)}个epoch，无法进行趋势分析（至少需要3个）。")


def print_worst_episodes(
    data: LoadedData,
    ranges: list[tuple[int, int]],
    continuous: np.ndarray,
    deadband: float,
    top_n: int,
) -> None:
    print_title("7. Policy平移动作最抖的完整Episode")
    rows = episode_rows(data, ranges, continuous, deadband)
    rows = [
        row
        for row in rows
        if row["policy_pairs"] >= 5 and np.isfinite(row["jump_mean"])
    ]
    rows.sort(key=lambda row: row["jump_mean"], reverse=True)
    rows = rows[: max(0, top_n)]
    if not rows:
        print("没有包含至少5个连续Policy动作对的完整Episode。")
        return

    print(
        f"{'Episode':>9}{'长度':>8}{'成功':>8}{'Policy步':>10}"
        f"{'人工占比':>12}{'Policy对':>10}"
        f"{'平移跳变均值':>16}{'P95':>10}{'反向率':>10}"
    )
    for row in rows:
        print(
            f"{row['episode']:>9d}{row['length']:>8d}"
            f"{('是' if row['success'] else '否'):>8}"
            f"{row['policy_steps']:>10d}"
            f"{(fmt(row['human_percent'], 1) + '%'):>12}"
            f"{row['policy_pairs']:>10d}"
            f"{fmt(row['jump_mean']):>16}"
            f"{fmt(row['jump_p95']):>10}"
            f"{(fmt(row['reversal'], 1) + '%'):>10}"
        )


def print_conclusion(
    data: LoadedData,
    indices: np.ndarray,
    categories: dict[str, np.ndarray],
) -> None:
    print_title("8. 数据结论")
    policy_indices = indices[categories["Policy→Policy"]]
    human_indices = indices[categories["人工→人工"]]
    policy_jump = np.linalg.norm(
        data.actions[policy_indices, :3]
        - data.actions[policy_indices - 1, :3],
        axis=1,
    )
    human_jump = np.linalg.norm(
        data.actions[human_indices, :3]
        - data.actions[human_indices - 1, :3],
        axis=1,
    )

    policy_mean = mean(policy_jump)
    human_mean = mean(human_jump)
    ratio = (
        policy_mean / human_mean
        if np.isfinite(policy_mean)
        and np.isfinite(human_mean)
        and human_mean > 0
        else float("nan")
    )
    print(
        f"连续Policy平移动作平均跳变: {fmt(policy_mean)}\n"
        f"连续人工平移动作平均跳变:   {fmt(human_mean)}\n"
        f"Policy/人工跳变比:           {fmt(ratio, 2)}倍"
    )
    if np.isfinite(ratio):
        if ratio >= 3.0:
            print("判断: Policy动作明显比人工动作更抖。")
        elif ratio >= 1.5:
            print("判断: Policy动作比人工动作更抖，但差距中等。")
        else:
            print("判断: Policy和人工动作跳变接近，应重点检查下游控制。")
    print(
        "\n限制: buffer没有保存每步时间戳和原始SpaceMouse读数，"
        "因此本脚本能分析“实际执行动作”和“末端结果”，"
        "不能直接判断控制周期抖动或SpaceMouse硬件噪声。"
    )


def print_reward_analysis(
    data: LoadedData,
    ranges: list[tuple[int, int]],
    continuous: np.ndarray,
) -> None:
    print_title("9. 奖励函数与抖动关系分析")

    # 1. 奖励分布统计
    print("\n--- 9.1 奖励分布 ---")
    policy_rewards = data.rewards[~data.human]
    human_rewards = data.rewards[data.human]
    all_rewards = data.rewards

    print(f"奖励样本: 总数={len(all_rewards)}, "
          f"Policy步={len(policy_rewards)}, 人工步={len(human_rewards)}")
    print(f"Reward 唯一值: {np.unique(all_rewards).tolist()}")
    for label, rw in [("全部", all_rewards), ("Policy步", policy_rewards), ("人工步", human_rewards)]:
        pos = int((rw > 0).sum())
        neg = int((rw < 0).sum())
        zero = int((rw == 0).sum())
        print(f"  {label}: >0={pos} ({fmt(100*pos/len(rw),1)}%), "
              f"<0={neg} ({fmt(100*neg/len(rw),1)}%), "
              f"==0={zero} ({fmt(100*zero/len(rw),1)}%), "
              f"均值={fmt(mean(rw), 4)}")

    # 2. 每个 episode 的奖励、抖动、人工占比汇总
    print("\n--- 9.2 Episode 级别: 奖励 vs 抖动 vs 人工占比 ---")
    rows = episode_rows(data, ranges, continuous, deadband=0.03)
    valid_rows = [r for r in rows if r["policy_pairs"] >= 3 and np.isfinite(r["jump_mean"])]

    if not valid_rows:
        print("有效 episode 数不足。")
        return

    jumps = np.array([r["jump_mean"] for r in valid_rows])
    reversals = np.array([r["reversal"] for r in valid_rows if np.isfinite(r["reversal"])])
    human_pcts = np.array([r["human_percent"] for r in valid_rows])
    successes = np.array([r["success"] for r in valid_rows])

    # 按抖动三分位分组
    jump_thresholds = np.percentile(jumps, [33, 67])
    low_jitter = jumps <= jump_thresholds[0]
    mid_jitter = (jumps > jump_thresholds[0]) & (jumps <= jump_thresholds[1])
    high_jitter = jumps > jump_thresholds[1]

    print(f"\n抖动三分位阈值: 低≤{fmt(jump_thresholds[0])} < 中≤{fmt(jump_thresholds[1])} < 高")
    print(f"{'抖动等级':<12}{'Episode数':>10}{'成功率':>10}{'平均人工占比':>14}{'平均抖动':>12}{'平均反向率':>12}")
    for label, mask in [("低抖动", low_jitter), ("中抖动", mid_jitter), ("高抖动", high_jitter)]:
        n = int(mask.sum())
        suc = float(np.mean(successes[mask])) * 100 if n > 0 else float("nan")
        hp = mean(human_pcts[mask])
        jm = mean(jumps[mask])
        rev = mean(reversals[mask]) if n > 0 else float("nan")
        print(f"{label:<12}{n:>10d}{fmt(suc,1) + '%':>10}"
              f"{fmt(hp,1) + '%':>14}{fmt(jm):>12}{fmt(rev,1) + '%':>12}")

    # 2b. 按人工占比分三组
    hp_thresholds = np.percentile(human_pcts, [33, 67])
    low_hp = human_pcts <= hp_thresholds[0]
    mid_hp = (human_pcts > hp_thresholds[0]) & (human_pcts <= hp_thresholds[1])
    high_hp = human_pcts > hp_thresholds[1]

    print(f"\n人工占比三分位阈值: 低≤{fmt(hp_thresholds[0],1)}% < 中≤{fmt(hp_thresholds[1],1)}% < 高")
    print(f"{'人工占比':<12}{'Episode数':>10}{'成功率':>10}{'平均抖动':>12}{'平均反向率':>12}")
    for label, mask in [("低人工占比", low_hp), ("中人工占比", mid_hp), ("高人工占比", high_hp)]:
        n = int(mask.sum())
        suc = float(np.mean(successes[mask])) * 100 if n > 0 else float("nan")
        jm = mean(jumps[mask])
        rev = mean(reversals[mask]) if n > 0 else float("nan")
        print(f"{label:<12}{n:>10d}{fmt(suc,1) + '%':>10}{fmt(jm):>12}{fmt(rev,1) + '%':>12}")

    # 3. 奖励事件分析
    print("\n--- 9.3 奖励事件分析 ---")
    pos_mask = data.rewards > 0
    if np.any(pos_mask):
        pos_indices = np.flatnonzero(pos_mask)
        pos_jumps = []
        pos_rev = []
        for idx in pos_indices:
            if idx > 0:
                prev = data.actions[idx - 1, :3]
                cur = data.actions[idx, :3]
                pos_jumps.append(np.linalg.norm(cur - prev))
                if np.linalg.norm(prev) > 0.03 and np.linalg.norm(cur) > 0.03:
                    pos_rev.append(float(np.sum(prev * cur) < 0))
        if pos_jumps:
            print(f"正奖励事件 (reward>0): {int(pos_mask.sum())} 次")
            print(f"  奖励前→奖励步 平移跳变 均值={fmt(mean(np.array(pos_jumps)))} "
                  f"P95={fmt(percentile(np.array(pos_jumps), 95))}")
        if pos_rev:
            print(f"  奖励步平移反向率: {fmt(100*mean(np.array(pos_rev)), 1)}%")

    # 4. 奖励分布变化趋势 (按 epoch)
    print("\n--- 9.4 奖励随训练变化趋势 ---")
    epoch_size = 50
    n_epochs = len(valid_rows) // epoch_size
    if n_epochs >= 3:
        print(f"{'Epoch':>8}{'Ep范围':>12}{'成功次数':>10}{'成功率':>10}"
              f"{'正奖励':>10}{'零奖励':>10}{'负奖励':>10}{'平均抖动':>12}")
        for e in range(n_epochs):
            ep_rows = valid_rows[e*epoch_size:(e+1)*epoch_size]
            # 统计这些 episode 在原数据中的位置
            start_idx = ranges[ep_rows[0]["episode"] - 1][0]
            end_idx = ranges[ep_rows[-1]["episode"] - 1][1]
            seg_rewards = data.rewards[start_idx:end_idx]
            pos = int((seg_rewards > 0).sum())
            neg = int((seg_rewards < 0).sum())
            zero = int((seg_rewards == 0).sum())
            suc = float(np.mean([r["success"] for r in ep_rows])) * 100
            jm = mean(np.array([r["jump_mean"] for r in ep_rows]))
            print(f"{e+1:>8d}{ep_rows[0]['episode']}-{ep_rows[-1]['episode']:>5}"
                  f"{int(sum([r['success'] for r in ep_rows])):>10d}"
                  f"{fmt(suc,1) + '%':>10}"
                  f"{pos:>10d}{zero:>10d}{neg:>10d}{fmt(jm):>12}")

    # 5. 平滑度惩罚分析 (仅新版5D数据)
    print("\n--- 9.5 平滑度惩罚分析 ---")
    is_7d = data.actions.shape[1] == 7
    if is_7d:
        print("旧版 7D 数据（无 last_action 观测），无平滑度惩罚 (smoothness_penalty)。")
        print("所有 reward ∈ {0, 1}，仅由脚踏板/分类器触发。")
    else:
        neg_rewards = data.rewards[data.rewards < 0]
        neg_frac = len(neg_rewards) / len(data.rewards) if len(data.rewards) > 0 else 0
        if neg_frac > 0:
            print(f"负奖励步数: {len(neg_rewards)}/{len(data.rewards)} ({fmt(100*neg_frac, 1)}%)")
            print(f"负奖励范围: [{fmt(float(np.min(neg_rewards)), 6)}, "
                  f"{fmt(float(np.max(neg_rewards)), 6)}]")
            print(f"负奖励均值: {fmt(mean(neg_rewards), 6)}")
            print("这些负奖励 = -smoothness_penalty = -0.001 * ||last_action(t) - last_action(t-1)||")
        else:
            print("未发现负奖励，smoothness_penalty 可能为 0 或被禁用。")

        zero_rewards = data.rewards[data.rewards == 0]
        pos_rewards = data.rewards[data.rewards > 0]
        print(f"奖励分布: 正={len(pos_rewards)} ({fmt(100*len(pos_rewards)/len(data.rewards),1)}%), "
              f"零={len(zero_rewards)} ({fmt(100*len(zero_rewards)/len(data.rewards),1)}%), "
              f"负={len(neg_rewards)} ({fmt(100*len(neg_rewards)/len(data.rewards),1)}%)")

    # 6. 跨数据集对比提示
    print("\n--- 9.6 奖励函数设计要点 ---")
    print("旧版 (7D, 无 last_action):")
    print("  奖励来源: 仅脚踏板 (Shift+←) → reward=1.0")
    print("  reward ∈ {0, 1}, 无平滑度惩罚")
    print("  问题: Policy 不知道自己在做什么, 抖动无法被惩罚")
    print()
    print("新版 (5D, 有 last_action):")
    print("  奖励来源: 分类器 + 脚踏板 → base_reward ∈ {0, 1}")
    print("          smoothness_penalty = 0.001 * ||last_action(t) - last_action(t-1)||")
    print("          final_reward = base_reward - smoothness_penalty")
    print("          (sigmoid > 1.70 禁用分类器, 仅用脚踏板)")
    print("  效果: 通过 last_action 差分惩罚, 鼓励平滑动作")


def main() -> int:
    args = parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    pairs = select_file_pairs(checkpoint_dir, args.last_n_files)
    if not pairs:
        print(
            f"没有找到可分析的成对分片: {checkpoint_dir}",
            file=sys.stderr,
        )
        return 2

    print(f"Checkpoint目录: {checkpoint_dir}")
    print(
        "正在逐个读取分片；全部数据可能较慢，"
        "快速检查可使用 --last-n-files 5。"
    )
    try:
        data = load_data(pairs)
    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        return 1

    if data.actions.ndim != 2 or data.actions.shape[1] < 5:
        print(
            f"Action shape异常: {data.actions.shape}",
            file=sys.stderr,
        )
        return 1

    # 根据数据自动选择动作名称
    global ACTION_NAMES
    if data.actions.shape[1] == 7:
        ACTION_NAMES = ACTION_NAMES_7D
        print(f"[检测] 发现 7D 动作空间 (旧版: dx,dy,dz,drx,dry,drz,gripper)")
    else:
        ACTION_NAMES = ACTION_NAMES_5D
        print(f"[检测] 发现 {data.actions.shape[1]}D 动作空间 (新版: dx,dy,dz,drz,gripper)")

    continuous = continuity_mask(data)
    ranges = print_dataset_summary(
        data,
        continuous,
        full_history=(args.last_n_files == 0),
    )
    print(
        f"可确认的连续Transition对: "
        f"{int(continuous.sum())}/{max(0, len(continuous) - 1)}"
    )
    print_action_size(data)
    indices, categories = print_action_jitter(
        data, continuous, args.deadband
    )
    print_observed_motion(data, indices, categories)
    print_gripper_summary(data)
    print_jitter_trend_by_epochs(
        data, ranges, continuous, args.deadband, epoch_size=100
    )
    print_worst_episodes(
        data,
        ranges,
        continuous,
        args.deadband,
        args.top_episodes,
    )
    print_conclusion(data, indices, categories)
    print_reward_analysis(data, ranges, continuous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
