#!/usr/bin/env python3
"""Render a buffer episode as a synchronized camera/state MP4.

The script is strictly offline: it reads pickle files and never creates or
steps a robot environment.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from visualize_demo_full import (
    ACTION_NAMES,
    STATE_NAMES,
    load_and_mark_human,
    load_latest_files,
    split_episodes,
)


CAMERA_KEYS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render camera images, actions and state traces from one buffer episode."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--buffer-file")
    parser.add_argument("--demo-file")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=-1,
        help="Complete episode index in the selected file; -1 selects the latest.",
    )
    parser.add_argument(
        "--output",
        default="utils/buffer_videos/latest_episode.mp4",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth transition (default: 1).",
    )
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def extract_image(observation, key):
    image = np.asarray(observation[key])
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unexpected {key} image shape: {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image[..., :3]


def padded_limits(values, minimum_span=0.1):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    span = max(high - low, minimum_span)
    return low - 0.08 * span, high + 0.08 * span


def add_trace_panel(fig, spec, title, steps, values, labels):
    ax = fig.add_subplot(spec)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, values.shape[1]))
    progress_lines = []
    for dim, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(steps, values[:, dim], color=color, alpha=0.14, linewidth=1.0)
        line, = ax.plot([], [], color=color, linewidth=1.8, label=label)
        progress_lines.append(line)
    cursor = ax.axvline(0, color="#111827", linewidth=1.1, alpha=0.8)
    ax.set_xlim(0, max(1, len(steps) - 1))
    ax.set_ylim(*padded_limits(values))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.22, linestyle="--")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best", ncol=2, framealpha=0.75)
    return ax, progress_lines, cursor, values


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    if args.buffer_file or args.demo_file:
        if not (args.buffer_file and args.demo_file):
            raise ValueError("--buffer-file and --demo-file must be provided together")
        buffer_file, demo_file = args.buffer_file, args.demo_file
    else:
        buffer_file, demo_file = load_latest_files(args.checkpoint_dir)

    transitions, is_human = load_and_mark_human(buffer_file, demo_file)
    episodes = split_episodes(transitions, is_human)
    if not episodes:
        raise RuntimeError("No complete episode found in the selected buffer file")

    episode_index = args.episode_index
    if not -len(episodes) <= episode_index < len(episodes):
        raise IndexError(
            f"episode index {episode_index} is outside "
            f"[-{len(episodes)}, {len(episodes) - 1}]"
        )
    episode = episodes[episode_index]
    indices = episode["indices"]
    episode_transitions = [transitions[i] for i in indices]
    human = np.asarray(episode["is_human"], dtype=bool)

    actions = np.asarray([t["actions"] for t in episode_transitions], dtype=np.float32)
    states = np.asarray(
        [np.asarray(t["observations"]["state"]).reshape(-1) for t in episode_transitions],
        dtype=np.float32,
    )
    rewards = np.asarray([t["rewards"] for t in episode_transitions], dtype=np.float32)
    if actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected {len(ACTION_NAMES)} actions, got {actions.shape[1]}")
    if states.shape[1] != len(STATE_NAMES):
        raise ValueError(f"Expected {len(STATE_NAMES)} states, got {states.shape[1]}")
    for key in CAMERA_KEYS:
        if key not in episode_transitions[0]["observations"]:
            raise KeyError(f"Camera {key!r} is missing from observations")

    steps = np.arange(len(episode_transitions))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22.0, 10.8), dpi=args.dpi, facecolor="#f5f7fa")
    grid = fig.add_gridspec(
        4,
        5,
        height_ratios=(1.1, 1.1, 1.0, 1.0),
        width_ratios=(1.0, 1.0, 1.0, 1.0, 1.05),
        hspace=0.43,
        wspace=0.28,
        left=0.035,
        right=0.985,
        top=0.93,
        bottom=0.055,
    )
    fig.suptitle(
        f"Buffer Episode | {os.path.basename(buffer_file)} | "
        f"episode={episode_index} | length={len(steps)}",
        fontsize=15,
        fontweight="bold",
    )

    image_artists = []
    image_axes = []
    first_obs = episode_transitions[0]["observations"]
    for col, key in enumerate(CAMERA_KEYS):
        ax = fig.add_subplot(grid[0:2, col])
        artist = ax.imshow(extract_image(first_obs, key), interpolation="nearest")
        ax.set_title(key, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        image_axes.append(ax)
        image_artists.append(artist)

    status_ax = fig.add_subplot(grid[0:2, 4])
    status_ax.axis("off")
    status_text = status_ax.text(
        0.01,
        0.99,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.2,
        linespacing=1.3,
    )

    panels = []
    panels.append(add_trace_panel(fig, grid[2, 0], "Translation action", steps, actions[:, 0:3], ACTION_NAMES[0:3]))
    panels.append(add_trace_panel(fig, grid[2, 1], "Yaw / gripper action", steps, actions[:, 3:5], ACTION_NAMES[3:5]))
    panels.append(add_trace_panel(fig, grid[2, 2], "TCP position (normalized)", steps, states[:, 9:12], ("x", "y", "z")))
    panels.append(add_trace_panel(fig, grid[2, 3], "TCP rotation (normalized)", steps, states[:, 12:15], ("rx", "ry", "rz")))
    panels.append(add_trace_panel(fig, grid[3, 0], "Force (normalized)", steps, states[:, 6:9], ("Fx", "Fy", "Fz")))
    panels.append(add_trace_panel(fig, grid[3, 1], "Torque (normalized)", steps, states[:, 15:18], ("Tx", "Ty", "Tz")))
    panels.append(add_trace_panel(fig, grid[3, 2], "Linear velocity (normalized)", steps, states[:, 18:21], ("Vx", "Vy", "Vz")))
    panels.append(add_trace_panel(fig, grid[3, 3], "Angular velocity (normalized)", steps, states[:, 21:24], ("Vrx", "Vry", "Vrz")))

    frame_indices = list(range(0, len(steps), args.frame_step))
    if frame_indices[-1] != len(steps) - 1:
        frame_indices.append(len(steps) - 1)

    def update(frame_number):
        step = frame_indices[frame_number]
        transition = episode_transitions[step]
        obs = transition["observations"]
        mode = "HUMAN INTERVENTION" if human[step] else "ROBOT"
        mode_color = "#2471A3" if human[step] else "#D35400"

        artists = []
        for key, ax, artist in zip(CAMERA_KEYS, image_axes, image_artists):
            artist.set_data(extract_image(obs, key))
            for spine in ax.spines.values():
                spine.set_edgecolor(mode_color)
            artists.append(artist)

        for _, lines, cursor, values in panels:
            for dim, line in enumerate(lines):
                line.set_data(steps[: step + 1], values[: step + 1, dim])
                artists.append(line)
            cursor.set_xdata([step, step])
            artists.append(cursor)

        action = actions[step]
        state = states[step]
        terminal_label = ""
        if rewards[step] >= 0.5:
            terminal_label = "SUCCESS"
        elif rewards[step] <= -0.5:
            terminal_label = "UNRECOVERABLE FAILURE"
        elif bool(transition["dones"]):
            terminal_label = "TIMEOUT / TERMINAL"
        status = [
            f"STEP     {step + 1:03d}/{len(steps):03d}",
            f"MODE     {mode}",
            f"REWARD   {rewards[step]: .4f}",
            f"RETURN   {rewards[:step + 1].sum(): .4f}",
            f"DONE     {bool(transition['dones'])}",
            f"RESULT   {terminal_label or '-'}",
            "",
            "ACTION",
            f" dx/dy/dz {action[0]: .3f} {action[1]: .3f} {action[2]: .3f}",
            f" drz/grip {action[3]: .3f} {action[4]: .3f}",
            "",
            "STATE (normalized)",
            f" gripper  {state[0]: .3f}",
            f" last xyz {state[1]: .3f} {state[2]: .3f} {state[3]: .3f}",
            f" last rz/g{state[4]: .3f} {state[5]: .3f}",
            f" force    {state[6]: .3f} {state[7]: .3f} {state[8]: .3f}",
            f" tcp xyz  {state[9]: .3f} {state[10]: .3f} {state[11]: .3f}",
            f" tcp rpy  {state[12]: .3f} {state[13]: .3f} {state[14]: .3f}",
            f" torque   {state[15]: .3f} {state[16]: .3f} {state[17]: .3f}",
            f" vel xyz  {state[18]: .3f} {state[19]: .3f} {state[20]: .3f}",
            f" vel rpy  {state[21]: .3f} {state[22]: .3f} {state[23]: .3f}",
        ]
        status_text.set_text("\n".join(status))
        status_text.set_color(mode_color)
        artists.append(status_text)
        return artists

    effective_fps = args.fps / args.frame_step
    writer = animation.FFMpegWriter(
        fps=effective_fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Marvin USB buffer episode"},
    )
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / effective_fps,
        blit=False,
    )

    print(f"\nRendering {len(frame_indices)} frames at {effective_fps:g} fps...")
    print(f"Output: {output}")
    movie.save(output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
