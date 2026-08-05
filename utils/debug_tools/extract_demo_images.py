#!/usr/bin/env python3
"""Randomly sample and visualize RGB images from a HIL-SERL demo pickle."""

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np


IMAGE_FIELDS = ("observations", "next_observations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample demo transitions and export their camera images."
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
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of transitions to sample without replacement (default: 12).",
    )
    sample_group.add_argument(
        "--all",
        action="store_true",
        help="Export every transition, matching the previous script behavior.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random sampling seed (default: 0).")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not create the camera-grid preview image.",
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


def add_preview_label(rgb, lines):
    tile = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    label_height = 18 * len(lines) + 4
    cv2.rectangle(tile, (0, 0), (tile.shape[1], label_height), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(
            tile,
            line,
            (4, 16 + 18 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)


def save_preview(transitions, selected_indices, locations, fields, destination):
    camera_columns = sorted(
        {
            (field, camera)
            for transition_index in selected_indices
            for field in fields
            for camera in transitions[transition_index][field]
            if camera != "state"
        }
    )
    if not camera_columns:
        raise ValueError("No camera images found for preview")

    rows = []
    tile_size = (128, 128)
    for transition_index in selected_indices:
        episode_index, episode_step = locations[transition_index]
        row = []
        for field, camera in camera_columns:
            observation = transitions[transition_index][field]
            if camera not in observation:
                tile = np.zeros((*tile_size, 3), dtype=np.uint8)
            else:
                rgb = unwrap_rgb(
                    observation[camera], f"transition {transition_index} {field}.{camera}"
                )
                tile = cv2.resize(rgb, tile_size, interpolation=cv2.INTER_AREA)
            row.append(
                add_preview_label(
                    tile,
                    (
                        f"t={transition_index} ep={episode_index} step={episode_step}",
                        f"{field}.{camera}",
                    ),
                )
            )
        rows.append(np.concatenate(row, axis=1))

    preview = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(destination), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {destination}")


def main():
    args = parse_args()
    demo_path = args.demo_path.resolve()
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)

    project_root = Path(__file__).resolve().parents[2]
    sample_suffix = "all" if args.all else f"random_{args.num_samples}_seed_{args.seed}"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "utils" / "demo_image_exports" / demo_path.stem / sample_suffix
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

    if not args.all and args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")

    locations = []
    episode_index = 0
    episode_step = 0
    for transition in transitions:
        locations.append((episode_index, episode_step))
        episode_step += 1
        if bool(transition["dones"]):
            episode_index += 1
            episode_step = 0
    completed_episode_count = episode_index

    if args.all:
        selected_indices = list(range(len(transitions)))
    else:
        sample_count = min(args.num_samples, len(transitions))
        rng = np.random.default_rng(args.seed)
        selected_indices = sorted(
            int(index) for index in rng.choice(len(transitions), size=sample_count, replace=False)
        )

    fields = IMAGE_FIELDS if args.field == "both" else (args.field,)
    manifest_path = output_dir / "manifest.csv"
    image_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=(
                "transition_index", "episode_index", "episode_step", "field",
                "camera", "height", "width", "reward", "done", "relative_path",
            ),
        )
        writer.writeheader()

        for transition_index in selected_indices:
            transition = transitions[transition_index]
            selected_episode_index, selected_episode_step = locations[transition_index]
            done = bool(transition["dones"])
            for field in fields:
                observation = transition[field]
                camera_names = sorted(key for key in observation if key != "state")
                for camera in camera_names:
                    label = f"transition {transition_index} {field}.{camera}"
                    rgb = unwrap_rgb(observation[camera], label)
                    relative_path = Path(field) / f"episode_{selected_episode_index:03d}" / camera / (
                        f"step_{selected_episode_step:04d}_transition_{transition_index:06d}.png"
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
                            "episode_index": selected_episode_index,
                            "episode_step": selected_episode_step,
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

    preview_path = output_dir / "preview.jpg"
    if not args.no_preview:
        save_preview(transitions, selected_indices, locations, fields, preview_path)

    print(f"Demo: {demo_path}")
    print(f"Transitions: {len(transitions)}")
    print(f"Completed episodes: {completed_episode_count}")
    print(f"Selected transitions ({len(selected_indices)}): {selected_indices}")
    print(f"Exported images: {image_count}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if not args.no_preview:
        print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
