#!/usr/bin/env python3
"""Extract every stored RGB image from a HIL-SERL demo pickle."""

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np


IMAGE_FIELDS = ("observations", "next_observations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export demo observations/next_observations as lossless PNG files."
    )
    parser.add_argument("demo_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: utils/demo_image_exports/<demo filename without .pkl>",
    )
    parser.add_argument(
        "--field",
        choices=("observations", "next_observations", "both"),
        default="both",
        help="Which stored image fields to export (default: both).",
    )
    return parser.parse_args()


def unwrap_rgb(image, label):
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label}: expected (H, W, 3) or (1, H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{label}: expected uint8, got {array.dtype}")
    return array


def main():
    args = parse_args()
    demo_path = args.demo_path.resolve()
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "utils" / "demo_image_exports" / demo_path.stem
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with demo_path.open("rb") as handle:
        transitions = pickle.load(handle)
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("Demo pickle must contain a non-empty transition list")

    fields = IMAGE_FIELDS if args.field == "both" else (args.field,)
    manifest_path = output_dir / "manifest.csv"
    image_count = 0
    episode_index = 0
    episode_step = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "transition_index", "episode_index", "episode_step", "field",
                "camera", "height", "width", "reward", "done", "relative_path",
            ),
        )
        writer.writeheader()

        for transition_index, transition in enumerate(transitions):
            done = bool(transition["dones"])
            for field in fields:
                observation = transition[field]
                camera_names = sorted(key for key in observation if key != "state")
                for camera in camera_names:
                    label = f"transition {transition_index} {field}.{camera}"
                    rgb = unwrap_rgb(observation[camera], label)
                    relative_path = Path(field) / f"episode_{episode_index:03d}" / camera / (
                        f"step_{episode_step:04d}_transition_{transition_index:06d}.png"
                    )
                    destination = output_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(
                        str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ):
                        raise OSError(f"Failed to write {destination}")
                    writer.writerow(
                        {
                            "transition_index": transition_index,
                            "episode_index": episode_index,
                            "episode_step": episode_step,
                            "field": field,
                            "camera": camera,
                            "height": rgb.shape[0],
                            "width": rgb.shape[1],
                            "reward": float(transition["rewards"]),
                            "done": int(done),
                            "relative_path": relative_path.as_posix(),
                        }
                    )
                    image_count += 1

            episode_step += 1
            if done:
                episode_index += 1
                episode_step = 0

    print(f"Demo: {demo_path}")
    print(f"Transitions: {len(transitions)}")
    print(f"Completed episodes: {episode_index}")
    print(f"Exported images: {image_count}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
