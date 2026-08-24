#!/usr/bin/env python3
"""先用 RESET_POSE 规划逆解轨迹，再保存实测关节角作为唯一起始构型。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERL_INFRA = PROJECT_ROOT / "serl_robot_infra"
for path in (PROJECT_ROOT, SERL_INFRA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def angular_error_deg(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    error = actual - expected
    error[3:] = (error[3:] + 180.0) % 360.0 - 180.0
    return error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="移动到当前 RESET_POSE 并标定关节起始位"
    )
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--arm", choices=("A", "B"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="覆盖已有标定文件")
    return parser.parse_args()


def save_calibration(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"{path} 已存在；确认重新标定后加 --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig
    from examples.experiments.marvin_usb_insertion.wrapper import MarvinUSBEnv

    args = parse_args()
    config = MarvinUSBEnvConfig()
    if args.robot_ip:
        config.ROBOT_IP = args.robot_ip
    if args.arm:
        config.ARM = args.arm
    config.REALSENSE_CAMERAS = {}
    config.USE_GRIPPER = False
    output = args.output or Path(config.RESET_JOINTS_PATH)
    if output.exists() and not args.force:
        raise SystemExit(f"{output} 已存在；确认重新标定后加 --force")

    print(f"机械臂: {config.ARM} @ {config.ROBOT_IP}")
    print(f"目标 RESET_POSE (mm, deg): {config.RESET_POSE.tolist()}")
    print("将在阻抗模式下通过 movLA 逆解轨迹移动，请先确认路径无障碍物。")
    if input("输入 MOVE 开始：").strip() != "MOVE":
        print("已取消。")
        return 1

    env = None
    try:
        env = MarvinUSBEnv(
            hz=config.HZ, fake_env=False, save_video=False, config=config
        )
        env._movla_to_xyzabc(
            np.asarray(config.RESET_POSE, dtype=np.float64),
            vel=config.RESET_MOVLA_VEL,
            acc=config.RESET_MOVLA_ACC,
        )
        time.sleep(1.0)

        samples = []
        for _ in range(10):
            env._update_currpos()
            samples.append(np.asarray(env.curr_joints, dtype=np.float64).copy())
            time.sleep(0.05)
        samples = np.asarray(samples)
        joints = np.median(samples, axis=0)
        spread = np.max(np.ptp(samples, axis=0))
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"读到无效关节角: {joints}")
        if spread > 0.2:
            raise RuntimeError(f"关节仍在运动（采样极差 {spread:.3f}°），拒绝保存")

        fk = env.kk.fk(joints=joints.tolist())
        measured_pose = np.asarray(env.kk.mat4x4_to_xyzabc(pose_mat=fk), dtype=np.float64)
        pose_error = angular_error_deg(
            measured_pose.copy(), np.asarray(config.RESET_POSE, dtype=np.float64)
        )
        if np.max(np.abs(pose_error[:3])) > 3.0 or np.max(np.abs(pose_error[3:])) > 2.0:
            raise RuntimeError(
                "起始位未到位，拒绝保存；"
                f"XYZABC 误差={pose_error.tolist()}"
            )

        payload = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "robot_ip": config.ROBOT_IP,
            "arm": config.ARM,
            "units": {"joints": "deg", "tcp_pose": ["mm", "deg"]},
            "reset_pose_xyzabc": np.asarray(config.RESET_POSE).tolist(),
            "measured_pose_xyzabc": measured_pose.tolist(),
            "pose_error_xyzabc": pose_error.tolist(),
            "joints_deg": joints.tolist(),
            "sample_spread_deg": float(spread),
        }
        save_calibration(output, payload, args.force)
        print(f"\n标定已保存: {output}")
        print(f"关节角 (deg): {np.array2string(joints, precision=6)}")
        print(f"XYZABC 误差: {np.array2string(pose_error, precision=4)}")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
