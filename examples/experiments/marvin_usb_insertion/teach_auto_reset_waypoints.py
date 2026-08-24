#!/usr/bin/env python3
"""使用与录制 demo 相同的 SpaceMouse 控制链示教自动复位关键点。"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
import termios
import tty
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERL_INFRA = PROJECT_ROOT / "serl_robot_infra"
if str(SERL_INFRA) not in sys.path:
    sys.path.insert(0, str(SERL_INFRA))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_KINE_CONFIG = (
    SERL_INFRA / "marvin_env" / "SDK_PYTHON" / "ccs_m6_40.MvKDCfg"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用录制 demo 的 SpaceMouse 控制方式示教 USB 自动复位关键点"
    )
    parser.add_argument("--robot-ip", default="192.168.14.190")
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--kine-config", type=Path, default=DEFAULT_KINE_CONFIG)
    parser.add_argument(
        "--drop-range-mm",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(20.0, 20.0, 0.0),
        help="以 drop_center 为中心的 XYZ 随机半区间，默认 ±20 ±20 ±0 mm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 JSON；默认保存到本脚本目录并带时间戳",
    )
    return parser.parse_args()


def angular_error_deg(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """返回 XYZABC 差值，姿态角差值折叠到 [-180, 180)。"""
    error = actual - expected
    error[3:] = (error[3:] + 180.0) % 360.0 - 180.0
    return error


def format_pose(pose: np.ndarray) -> str:
    return " ".join(f"{value:9.3f}" for value in pose)


def command_gripper(env, command: str | None) -> None:
    if command is None:
        return
    if env.gripper is None:
        raise RuntimeError("夹爪未初始化，不能完成关键点示教流程")
    succeeded = getattr(env.gripper, command)(blocking=True)
    if not succeeded:
        raise RuntimeError(f"夹爪{command}失败")
    env._update_currpos()


def capture_waypoints(controller, env) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """连续执行 demo 控制，Enter 只采集当前 XYZABC。"""
    stages = (
        ("initial_pose", "初始位置", "移动到每次 episode 开始的位置。", None),
        ("usb_above", "点1/4", "移动到 U 盘/光模块正上方；保存后自动开夹爪。", "open"),
        (
            "usb_grasp",
            "点2/4",
            "下落到 U 盘两侧；确认手已离开，保存后自动闭合夹爪。",
            "close",
        ),
        ("extract_retreat", "点3/4", "后退到光模块完全安全抽出的位置。", None),
        (
            "drop_center",
            "点4/4",
            "移动到随机投放区中心；保存后自动开夹爪。",
            "open",
        ),
        ("return_check", "回位校验", "移动回刚才记录的初始位置。", None),
    )
    waypoints: dict[str, np.ndarray] = {}
    stage_index = 0
    old_terminal = termios.tcgetattr(sys.stdin)
    last_print = 0.0

    print("\n空间鼠标控制与录制 demo 一致：XYZ + 单轴旋转。")
    print("空间鼠标左键=关闭夹爪，右键=打开夹爪；键盘 Enter=保存当前点，Q=退出。")
    print(f"\n[{stages[0][1]}] {stages[0][2]}")
    try:
        tty.setcbreak(sys.stdin.fileno())
        while stage_index < len(stages):
            controller.step(np.zeros(env.action_space.shape, dtype=np.float32))
            now = time.monotonic()
            if now - last_print >= 1.0:
                env._update_currpos()
                print(f"\rTCP XYZABC: {format_pose(env.currpos_xyzabc)}", end="", flush=True)
                last_print = now

            if not select.select([sys.stdin], [], [], 0.0)[0]:
                continue
            key = sys.stdin.read(1)
            if key.lower() == "q":
                raise KeyboardInterrupt
            if key not in ("\r", "\n"):
                continue

            name, label, _instruction, gripper_command = stages[stage_index]
            env._update_currpos()
            pose = np.asarray(env.currpos_xyzabc, dtype=np.float64).copy()
            if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                raise RuntimeError(f"无效 TCP 位姿: {pose}")
            print(f"\n[{label}] 已记录: {format_pose(pose)}")
            command_gripper(env, gripper_command)
            if name == "return_check":
                return waypoints, pose
            waypoints[name] = pose
            stage_index += 1
            print(f"\n[{stages[stage_index][1]}] {stages[stage_index][2]}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal)

    raise RuntimeError("关键点采集流程异常结束")


def save_waypoints(
    path: Path,
    args: argparse.Namespace,
    waypoints: dict[str, np.ndarray],
    return_check: np.ndarray,
) -> None:
    error = angular_error_deg(return_check.copy(), waypoints["initial_pose"])
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "robot_ip": args.robot_ip,
        "arm": args.arm,
        "units": {"position": "mm", "orientation": "deg"},
        "drop_random_half_range_mm": list(map(float, args.drop_range_mm)),
        "waypoints": {name: pose.tolist() for name, pose in waypoints.items()},
        "return_check_pose": return_check.tolist(),
        "return_error_xyzabc": error.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if any(value < 0 for value in args.drop_range_mm):
        raise SystemExit("--drop-range-mm 必须是三个非负半区间")
    if not args.kine_config.is_file():
        raise SystemExit(f"运动学配置不存在: {args.kine_config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(__file__).with_name(
        f"auto_reset_waypoints_{args.arm}_{timestamp}.json"
    )
    if output.exists():
        raise SystemExit(f"输出文件已存在，拒绝覆盖: {output}")
    env = None
    expert = None

    try:
        from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig
        from franka_env.envs.relative_env import RelativeFrame
        from franka_env.envs.wrappers import SpacemouseIntervention
        from franka_env.utils.transformations import construct_transform_matrix
        from marvin_env.envs.marvin_env import MarvinEnv

        config = MarvinUSBEnvConfig()
        config.ROBOT_IP = args.robot_ip
        config.ARM = args.arm
        config.GRIPPER_MOTOR_ID = args.motor_id
        config.KINE_CONFIG_PATH = str(args.kine_config)
        config.REALSENSE_CAMERAS = {}
        config.DISPLAY_IMAGE = False
        config.MAX_EPISODE_LENGTH = 2**31 - 1
        config.TERMINATE_ON_POSE_SUCCESS = False

        print(f"连接 {args.robot_ip}，使用录制 demo 的控制链示教 {args.arm} 臂。")
        env = MarvinEnv(hz=config.HZ, fake_env=False, save_video=False, config=config)
        env._update_currpos()
        env._reset_stream_reference()
        # demo 在 reset 后启动，流式参考天然等于当前姿态；本脚本不自动移动，
        # 因此必须显式以当前实测姿态为参考，避免第一条零动作拉向 RESET_POSE。
        env._stream_ref_xyzabc = env.currpos_xyzabc.copy()

        relative_env = RelativeFrame(env, include_relative_pose=False)
        relative_env.transform_matrix = construct_transform_matrix(env.currpos_quat)
        controller = SpacemouseIntervention(relative_env)
        expert = controller.expert
        time.sleep(0.5)

        waypoints, return_check = capture_waypoints(controller, env)
        save_waypoints(output, args, waypoints, return_check)

        error = angular_error_deg(return_check.copy(), waypoints["initial_pose"])
        print(f"\n已保存: {output}")
        print(f"回位误差 XYZABC: {format_pose(error)}")
        print(f"随机投放半区间 XYZ(mm): {list(args.drop_range_mm)}")
        return 0
    except KeyboardInterrupt:
        print("\n已取消；未保存不完整的关键点文件。")
        return 130
    finally:
        if expert is not None:
            try:
                expert.close()
            except Exception as exc:
                print(f"SpaceMouse 关闭失败: {exc}", file=sys.stderr)
        if env is not None:
            try:
                if env.gripper is not None:
                    env.gripper.shutdown()
                env.close_cameras()
                env.robot.clear_set()
                env.robot.set_state(arm=args.arm, state=0)
                env.robot.send_cmd()
                time.sleep(0.3)
                env.robot.release_robot()
                print("机器人已安全下使能并断开。")
            except Exception as exc:
                print(f"清理连接失败，请人工确认机器人状态: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
