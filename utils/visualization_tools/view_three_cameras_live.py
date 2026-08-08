#!/usr/bin/env python3
"""Continuously display the three Marvin policy cameras.

This utility initializes cameras only. It never creates a MarvinEnv, connects
to the robot, or sends robot/gripper commands.
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "serl_robot_infra"))

from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture


PHYSICAL_CAMERA_NAMES = ("side_policy", "wrist_1", "wrist_2")
VIEW_SOURCES = OrderedDict(
    (
        ("side_policy", "side_policy"),
        ("wrist_1", "wrist_1"),
        ("wrist_1buchong", "wrist_1"),
        ("wrist_2", "wrist_2"),
    )
)
WINDOW_NAME = "Marvin USB | 3 physical cameras / 4 policy views"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continuously display the three Marvin policy cameras."
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=420,
        help="Displayed width/height of each camera tile (default: 420).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Display full raw frames instead of the crop used by the policy.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="utils/visualization_tools/camera_snapshots",
        help="Directory used when the S key is pressed.",
    )
    return parser.parse_args()


def letterbox(frame, size):
    """Resize without distorting aspect ratio and pad to a square tile."""
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - resized_width) // 2
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def process_frame(frame, camera_name, config, no_crop):
    if frame is None:
        raise RuntimeError(f"{camera_name}: received an empty frame")
    if no_crop:
        return frame
    crop = config.IMAGE_CROP.get(camera_name)
    cropped = crop(frame) if crop is not None else frame
    if cropped.size == 0:
        raise RuntimeError(
            f"{camera_name}: IMAGE_CROP produced an empty image from {frame.shape}"
        )
    # Match MarvinEnv._get_im exactly. RSCapture returns BGR, and cv2 displays
    # BGR, so no channel conversion is needed in this viewer.
    return cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_LINEAR)


def decorate_tile(frame, camera_name, tile_size, source_shape, processed_shape):
    tile = letterbox(frame, tile_size)
    header_height = 58
    canvas = np.zeros((tile_size + header_height, tile_size, 3), dtype=np.uint8)
    canvas[header_height:] = tile
    cv2.putText(
        canvas,
        camera_name,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    shape_text = (
        f"raw {source_shape[1]}x{source_shape[0]}  "
        f"view {processed_shape[1]}x{processed_shape[0]}"
    )
    cv2.putText(
        canvas,
        shape_text,
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_snapshots(frames, snapshot_dir):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved = []
    for camera_name, frame in frames.items():
        path = snapshot_dir / f"{camera_name}_{timestamp}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to save {path}")
        saved.append(path)
    return saved


def main():
    args = parse_args()
    if args.display_size < 128:
        raise ValueError("--display-size must be at least 128")

    config = MarvinUSBEnvConfig()
    captures = OrderedDict()
    snapshot_dir = (PROJECT_ROOT / args.snapshot_dir).resolve()

    print("=" * 72)
    print("Marvin three-camera live viewer (camera only; no robot control)")
    print("=" * 72)
    print("Do not run this together with run_actor.sh/run_test.sh: RealSense devices")
    print("can only be owned by one process at a time.\n")

    try:
        for camera_name in PHYSICAL_CAMERA_NAMES:
            camera_config = dict(config.REALSENSE_CAMERAS[camera_name])
            print(
                f"Opening {camera_name}: serial={camera_config['serial_number']} "
                f"dim={camera_config['dim']} fps={camera_config.get('fps', 15)} "
                f"exposure={camera_config.get('exposure', 40000)}"
            )
            capture = VideoCapture(
                RSCapture(name=camera_name, **camera_config), name=camera_name
            )
            captures[camera_name] = capture

        print("\nControls: q/Esc=quit, s=save three processed snapshots")
        print("View: " + ("raw full frame" if args.no_crop else "policy crop resized to 128x128"))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            args.display_size * len(VIEW_SOURCES),
            args.display_size + 96,
        )

        last_time = time.perf_counter()
        smoothed_fps = 0.0
        while True:
            # Read every physical RealSense exactly once per display iteration.
            raw_frames = OrderedDict(
                (camera_name, capture.read())
                for camera_name, capture in captures.items()
            )
            processed_frames = OrderedDict()
            tiles = []
            for view_name, source_name in VIEW_SOURCES.items():
                raw = raw_frames[source_name]
                processed = process_frame(raw, view_name, config, args.no_crop)
                processed_frames[view_name] = processed
                tiles.append(
                    decorate_tile(
                        processed,
                        view_name,
                        args.display_size,
                        raw.shape,
                        processed.shape,
                    )
                )

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - last_time, 1e-6)
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )
            last_time = now

            mosaic = np.concatenate(tiles, axis=1)
            footer = np.zeros((38, mosaic.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                footer,
                f"display FPS {smoothed_fps:5.1f} | q/Esc quit | s snapshot",
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (120, 255, 160),
                1,
                cv2.LINE_AA,
            )
            display = np.concatenate([mosaic, footer], axis=0)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\nExit requested.")
                break
            if key == ord("s"):
                saved = save_snapshots(processed_frames, snapshot_dir)
                print("Saved snapshots:")
                for path in saved:
                    print(f"  {path}")

            # Closing the GUI window should also end the process.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("\nWindow closed.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.")
    except Exception as exc:
        print(f"\nCamera viewer failed: {exc}", file=sys.stderr)
        return 1
    finally:
        print("Closing cameras...")
        for capture in captures.values():
            try:
                capture.close()
            except Exception as exc:
                print(f"Warning: camera close failed: {exc}", file=sys.stderr)
        cv2.destroyAllWindows()
        print("Cameras closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
