#!/usr/bin/env python3
"""Render random complete buffer episodes and concatenate them into one MP4."""

import argparse
import pickle
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from visualize_demo_full import split_episodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def shard_step(path):
    match = re.search(r"transitions_(\d+)\.pkl$", path.name)
    return int(match.group(1)) if match else -1


def discover_candidates(checkpoint_dir, count, rng):
    buffer_dir = checkpoint_dir / "buffer"
    demo_dir = checkpoint_dir / "demo_buffer"
    shards = sorted(buffer_dir.glob("transitions_*.pkl"), key=shard_step)
    if not shards:
        raise FileNotFoundError(f"No buffer shards found in {buffer_dir}")

    # Randomize shard order so the selected episodes are distributed across
    # the saved online buffer, while stopping once there is a healthy pool.
    rng.shuffle(shards)
    candidates = []
    scanned = []
    target_pool_size = max(count * 2, count)
    for buffer_file in shards:
        demo_file = demo_dir / buffer_file.name
        if not demo_file.is_file():
            continue
        with buffer_file.open("rb") as handle:
            transitions = pickle.load(handle)
        episodes = split_episodes(
            transitions, np.zeros(len(transitions), dtype=bool)
        )
        scanned.append((buffer_file.name, len(episodes)))
        candidates.extend(
            (buffer_file, demo_file, episode_index, len(episode["indices"]))
            for episode_index, episode in enumerate(episodes)
        )
        del transitions
        if len(candidates) >= target_pool_size:
            break

    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} complete episodes found after scanning shards"
        )
    return candidates, scanned


def main():
    args = parse_args()
    if args.count <= 0 or args.fps <= 0 or args.dpi <= 0:
        raise ValueError("count, fps and dpi must be positive")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    candidates, scanned = discover_candidates(checkpoint_dir, args.count, rng)
    selected = rng.sample(candidates, args.count)

    renderer = Path(__file__).resolve().parent / "visualize_buffer_video.py"
    with tempfile.TemporaryDirectory(prefix="marvin_episode_video_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        clips = []
        for clip_index, (buffer_file, demo_file, episode_index, length) in enumerate(
            selected, start=1
        ):
            clip = temp_dir / f"episode_{clip_index:02d}.mp4"
            print(
                f"\n[{clip_index}/{args.count}] {buffer_file.name} "
                f"episode={episode_index} length={length}",
                flush=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(renderer),
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--buffer-file", str(buffer_file),
                    "--demo-file", str(demo_file),
                    "--episode-index", str(episode_index),
                    "--output", str(clip),
                    "--fps", str(args.fps),
                    "--dpi", str(args.dpi),
                ],
                check=True,
            )
            clips.append(clip)

        concat_file = temp_dir / "clips.txt"
        concat_file.write_text(
            "".join(f"file '{clip.as_posix()}'\n" for clip in clips),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )

    print("\nScanned shards:")
    for shard, episode_count in scanned:
        print(f"  {shard}: {episode_count} complete episodes")
    print("Selected episodes:")
    for buffer_file, _, episode_index, length in selected:
        print(f"  {buffer_file.name}: episode={episode_index}, length={length}")
    print(f"Saved: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
