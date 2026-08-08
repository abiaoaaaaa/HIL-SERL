#!/usr/bin/env python3
"""Render random transitions from buffer shards as a four-camera MP4."""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = ("side_policy", "wrist_1buchong", "wrist_1", "wrist_2")
TILE_SIZE = 384
FOOTER_HEIGHT = 120


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("buffer_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds-per-sample", type=float, default=1.0)
    return parser.parse_args()


def unwrap_rgb(value, label):
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{label}: unexpected image shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def load_random_samples(shards, count, rng):
    # Buffer shards are normally equal-sized 1000-step chunks. Selecting a
    # shard uniformly and then a transition uniformly samples the full saved
    # buffer without holding several large pickle files in memory at once.
    selected_shards = rng.choice(len(shards), size=count, replace=True)
    samples = []
    for shard_index in sorted(set(int(index) for index in selected_shards)):
        path = shards[shard_index]
        with path.open("rb") as handle:
            transitions = pickle.load(handle)
        slots = np.flatnonzero(selected_shards == shard_index)
        local_indices = rng.choice(len(transitions), size=len(slots), replace=False)
        for slot, local_index in zip(slots, local_indices):
            transition = transitions[int(local_index)]
            observation = transition["observations"]
            images = {
                key: unwrap_rgb(observation[key], f"{path.name}:{local_index}:{key}").copy()
                for key in CAMERA_KEYS
            }
            samples.append(
                (
                    int(slot),
                    path.name,
                    int(local_index),
                    images,
                    np.asarray(transition["actions"], dtype=np.float32).reshape(-1),
                    float(transition["rewards"]),
                    bool(transition["dones"]),
                )
            )
        del transitions
    samples.sort(key=lambda item: item[0])
    return samples


def render_frame(sample, sample_number, total):
    _, shard_name, local_index, images, action, reward, done = sample
    width = TILE_SIZE * 2
    height = TILE_SIZE * 2 + FOOTER_HEIGHT
    frame = np.full((height, width, 3), 242, dtype=np.uint8)

    for camera_index, key in enumerate(CAMERA_KEYS):
        row, col = divmod(camera_index, 2)
        tile = cv2.resize(
            cv2.cvtColor(images[key], cv2.COLOR_RGB2BGR),
            (TILE_SIZE, TILE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.rectangle(tile, (0, 0), (TILE_SIZE, 34), (0, 0, 0), -1)
        cv2.putText(
            tile, key, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        y0, x0 = row * TILE_SIZE, col * TILE_SIZE
        frame[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = tile

    y = TILE_SIZE * 2 + 34
    cv2.putText(
        frame,
        f"random sample {sample_number}/{total} | {shard_name} | local transition {local_index}",
        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "action [dx dy dz drz grip] = " + " ".join(f"{value:+.3f}" for value in action),
        (12, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"reward={reward:+.4f} | done={done}",
        (12, y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 20, 20), 1, cv2.LINE_AA,
    )
    return frame


def main():
    args = parse_args()
    if args.count <= 0 or args.fps <= 0 or args.seconds_per_sample <= 0:
        raise ValueError("count, fps and seconds-per-sample must be positive")
    shards = sorted(args.buffer_dir.resolve().glob("transitions_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"No transitions_*.pkl in {args.buffer_dir}")

    rng = np.random.default_rng(args.seed)
    samples = load_random_samples(shards, args.count, rng)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (TILE_SIZE * 2, TILE_SIZE * 2 + FOOTER_HEIGHT)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    repeats = max(1, round(args.fps * args.seconds_per_sample))
    try:
        for index, sample in enumerate(samples, start=1):
            frame = render_frame(sample, index, len(samples))
            for _ in range(repeats):
                writer.write(frame)
    finally:
        writer.release()

    print(f"Buffer shards: {len(shards)}")
    print(f"Random seed: {args.seed}")
    print("Selected samples:")
    for _, shard, local_index, _, _, reward, done in samples:
        print(f"  {shard}:{local_index} reward={reward:+.4f} done={done}")
    print(f"Saved: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
