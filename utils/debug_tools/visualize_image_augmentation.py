#!/usr/bin/env python3
"""Render deterministic previews of a named SERL image augmentation mode."""

import argparse
import sys
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "serl_launcher"))

from serl_launcher.vision.data_augmentations import (  # noqa: E402
    IMAGE_AUGMENTATION_PRESETS,
    batched_image_augmentation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input RGB camera image (read with OpenCV).")
    parser.add_argument("--mode", choices=IMAGE_AUGMENTATION_PRESETS, default="medium")
    parser.add_argument("--count", type=int, default=8, help="Number of augmented variants.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/marvin_augmentation_preview.jpg"))
    return parser.parse_args()


def add_label(rgb_image: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (128, 20), (0, 0, 0), -1)
    cv2.putText(bgr, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    rgb = cv2.cvtColor(cv2.resize(bgr, (128, 128)), cv2.COLOR_BGR2RGB)
    batch = jnp.asarray(rgb, dtype=jnp.uint8)[None, None]
    tiles = [add_label(rgb, "original")]

    for seed in range(args.count):
        augmented = batched_image_augmentation(
            batch,
            jax.random.PRNGKey(seed),
            mode=args.mode,
            num_batch_dims=2,
        )
        tiles.append(add_label(np.asarray(augmented[0, 0]), f"{args.mode} seed={seed}"))

    columns = 3
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i : i + columns], axis=1) for i in range(0, len(tiles), columns)]
    grid = np.concatenate(rows, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.mode} augmentation preview to {args.output}")


if __name__ == "__main__":
    main()
