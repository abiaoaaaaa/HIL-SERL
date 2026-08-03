#!/usr/bin/env python3
"""Replay one sampled Marvin action trajectory without cameras or observations.

This is a standalone real-robot debugging program.  It deliberately does not
call the actor, learner, reward function, camera stack, or ``env.step()``.  It
does reuse the task config and Marvin's low-level controller so that tool/gravity
parameters, reset mode switching, impedance parameters, safety clipping,
action scaling, trajectory planning, gripper control, and 10 Hz substeps match
the training environment.

The executed pipeline is:

    sampled EE action -> continuous EMA -> RelativeFrame-equivalent transform
    -> action-space clip -> HZ-rate policy target -> 200 Hz Cartesian streaming
    reference -> Marvin robot

Ctrl+C uses MarvinEnv's registered emergency-stop handler.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SERL_INFRA = PROJECT_ROOT / "serl_robot_infra"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
for path in (PROJECT_ROOT, SERL_INFRA, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig  # noqa: E402
from examples.experiments.marvin_usb_insertion.wrapper import (  # noqa: E402
    MarvinUSBEnv,
)
from fx_kine import FX_InvKineSolvePara  # noqa: E402


DEFAULT_SAMPLE_PATH = str(
    PROJECT_ROOT
    / "examples/demo_data/marvin_usb_insertion_20_demos_2026-07-28_09-47-54.pkl"
)
ACTION_NAMES = ("dx", "dy", "dz", "drz", "gripper")


# =============================================================================
# STANDALONE CONTROL TUNING PARAMETERS
# Edit values in this section directly.  They no longer come from TrainConfig or
# MarvinUSBEnvConfig at runtime.  Defaults below intentionally match the current
# Marvin USB training configuration.
# =============================================================================

DEFAULT_EMA_ALPHA = 0.65

# The policy runs at 10 Hz, but the reference sent to the impedance controller
# is generated continuously.  A policy action only updates the latest Cartesian
# goal; it never causes a new movLA trajectory to be planned and compressed.
CARTESIAN_SEND_HZ = 200
CARTESIAN_SUBSTEP_SECONDS = 0.05

# Reference-generator limits.  These are physical values, not percentages.
# Keep them independent of ACTION_SCALE: the latter is part of the policy's
# action semantics, while these limits determine how gently the reference can
# move toward the latest policy goal.
STREAM_LINEAR_VEL_MM_S = 60.0
STREAM_LINEAR_ACC_MM_S2 = 200.0
STREAM_YAW_VEL_DEG_S = 20.0
STREAM_YAW_ACC_DEG_S2 = 120.0


class StandaloneControlConfig(MarvinUSBEnvConfig):
    """Motion-related configuration owned by this standalone program."""

    # Robot and kinematics
    ROBOT_IP = "192.168.14.190"
    ARM = "A"
    KINE_CONFIG_PATH = str(
        PROJECT_ROOT
        / "serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"
    )

    # No cameras/observations in this program.
    REALSENSE_CAMERAS = {}
    IMAGE_CROP = {}
    DISPLAY_IMAGE = False

    # Actor action scaling: [translation mm, rotation rad, gripper].
    # The full action updates one policy-rate target; the shared MarvinEnv
    # streaming controller performs the 200 Hz interpolation.
    ACTION_SCALE = np.array([10.0, 0.05, 1.0], dtype=np.float64)
    HZ = 10

    # Reset pose and absolute Cartesian safety limits: [mm, mm, mm, deg, deg, deg].
    RESET_POSE = np.array(
        [394.3, 321.7, 200.3, -90.0005, 0.001, -90.001], dtype=np.float64
    )
    ABS_POSE_LIMIT_LOW = np.array(
        [325.8, 274.1, 123.1, -111.9, -11.9, -99.0], dtype=np.float64
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [518.6, 400.3, 405.7, -79.2, 12.8, -76.6], dtype=np.float64
    )
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 8.0
    RANDOM_RZ_RANGE = 0.08
    JOINT_RESET_PERIOD = 0

    # MarvinEnv always locks A/B to RESET_POSE during action execution and lets
    # C rotate. FIXED_ORIENTATION=True additionally suppresses policy drz.
    FIXED_ORIENTATION = False
    FIXED_ORIENTATION_ABC = None

    # Main control-mode selector: "cartesian" or "joint".
    IMPEDANCE_MODE = "cartesian"

    # Cartesian compliance used while replaying actor actions (Kp-like stiffness
    # and damping ratios). These values have a large effect on contact response.
    COMPLIANCE_PARAM = {
        "K": np.array(
            [6000.0, 6000.0, 6000.0, 600.0, 600.0, 600.0, 20.0],
            dtype=np.float64,
        ),
        "D": np.array(
            [0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0], dtype=np.float64
        ),
    }

    # High-stiffness mode used only while moving to RESET_POSE.
    PRECISION_PARAM = {
        "K": np.array(
            [4000.0, 4000.0, 4000.0, 600.0, 600.0, 600.0, 20.0],
            dtype=np.float64,
        ),
        "D": np.array(
            [0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0], dtype=np.float64
        ),
    }

    # Used only when IMPEDANCE_MODE="joint".
    JOINT_COMPLIANCE_PARAM = {
        "K": np.array([12.0, 12.0, 12.0, 10.0, 9.0, 9.0, 7.0], dtype=np.float64),
        "D": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2], dtype=np.float64),
    }

    # Ratios sent when entering/switching impedance modes. The standalone
    # Cartesian substep below plans and streams at the module-level rates.
    VEL_RATIO = 100
    ACC_RATIO = 100

    # Tool transform and dynamics used by set_tool() for gravity compensation.
    # Dynamic format: [mass kg, COM x/y/z mm, Ixx/Ixy/Ixz/Iyy/Iyz/Izz].
    TOOL_KINE_PARAMS = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    TOOL_DYN_PARAMS = np.array(
        [1.0, 0.0, 0.0, 50.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.03],
        dtype=np.float64,
    )

    # Gripper and episode reset behavior.
    USE_GRIPPER = True
    GRIPPER_MOTOR_ID = 1
    GRIPPER_SLEEP = 0.2
    AUTO_RESET_USB = False
    MANUAL_RESET_TIMEOUT = 8.0


class ContinuousActionEMA:
    """Same continuous-only EMA used by the RLPD actor."""

    def __init__(self, alpha: float, continuous_dims: int = 4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.continuous_dims = continuous_dims
        self.previous = np.zeros(continuous_dims, dtype=np.float32)

    def reset(self):
        self.previous.fill(0.0)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action, dtype=np.float32).copy()
        continuous = result[: self.continuous_dims]
        continuous = self.alpha * continuous + (1.0 - self.alpha) * self.previous
        result[: self.continuous_dims] = continuous
        self.previous = continuous.copy()
        return result


class NoObservationMarvinUSBEnv(MarvinUSBEnv):
    """Marvin USB controller with camera acquisition disabled."""

    def init_cameras(self, name_serial_dict=None):
        self.cap = {}
        print("[standalone] Cameras disabled; no observations will be acquired.")

    def reset_without_observation(self):
        """Match MarvinUSBEnv.reset() motion/mode behavior without _get_obs()."""
        print("\n[standalone] Resetting robot without observations...")
        self._recover()
        self._update_currpos()
        time.sleep(0.1)

        if getattr(self.config, "AUTO_RESET_USB", False):
            self._auto_reset_usb()
        else:
            self._manual_reset_usb()

        self.curr_path_length = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        self.last_joints = None
        self._update_currpos()
        self._reset_stream_reference()
        print("[standalone] Reset complete; compliance mode is active.\n")

    def _reset_stream_reference(self):
        """Start the streaming reference at the measured robot state.

        The reference pose, its Cartesian velocity, and the IK seed are kept
        across policy actions.  This is the essential difference from the old
        implementation, which planned a fresh rest-to-rest movLA segment for
        every 50 ms substep.
        """
        self._stream_ref_xyzabc = self.currpos_xyzabc.copy()
        self._stream_ref_xyzabc[3] = self._RESET_POSE[3]
        self._stream_ref_xyzabc[4] = self._RESET_POSE[4]
        self._stream_linear_velocity = np.zeros(3, dtype=np.float64)
        self._stream_yaw_velocity = 0.0
        self._stream_ref_joints = self.curr_joints.copy()

    @staticmethod
    def _step_vector_reference(position, velocity, goal, max_velocity, max_acceleration, dt):
        """Advance a 3-D velocity-limited, acceleration-limited reference."""
        error = goal - position
        distance = np.linalg.norm(error)
        if distance < 1e-9:
            desired_velocity = np.zeros(3, dtype=np.float64)
        else:
            # Braking-distance limit makes the reference settle at the goal
            # instead of crossing it at the configured maximum velocity.
            stopping_speed = np.sqrt(2.0 * max_acceleration * distance)
            desired_speed = min(max_velocity, stopping_speed)
            desired_velocity = error * (desired_speed / distance)

        delta_velocity = desired_velocity - velocity
        delta_norm = np.linalg.norm(delta_velocity)
        max_delta = max_acceleration * dt
        if delta_norm > max_delta:
            delta_velocity *= max_delta / delta_norm
        velocity = velocity + delta_velocity

        next_position = position + velocity * dt
        if np.dot(goal - position, goal - next_position) <= 0.0:
            return goal.copy(), np.zeros(3, dtype=np.float64)
        return next_position, velocity

    @staticmethod
    def _step_scalar_reference(position, velocity, goal, max_velocity, max_acceleration, dt):
        """Scalar counterpart used for the permitted C/Yaw rotation."""
        error = goal - position
        if abs(error) < 1e-9:
            desired_velocity = 0.0
        else:
            stopping_speed = np.sqrt(2.0 * max_acceleration * abs(error))
            desired_velocity = np.copysign(min(max_velocity, stopping_speed), error)

        velocity += np.clip(
            desired_velocity - velocity,
            -max_acceleration * dt,
            max_acceleration * dt,
        )
        next_position = position + velocity * dt
        if (goal - position) * (goal - next_position) <= 0.0:
            return float(goal), 0.0
        return float(next_position), float(velocity)

    def _stream_ik(self, xyzabc):
        """Solve one streaming reference pose using the prior IK solution."""
        ik_input = FX_InvKineSolvePara()
        pose_matrix = self.kk.xyzabc_to_mat4x4(xyzabc.tolist())
        if not pose_matrix:
            return None
        ik_input.set_all_inputs(
            target_tcp=self.kk.mat4x4_to_mat1x16(pose_matrix),
            ref_joint=self._stream_ref_joints.tolist(),
            zsp_type=0,
            zsp_para=[0.0] * 6,
            zsp_angle=0.0,
        )
        result = self.kk.ik(ik_input)
        if not result:
            return None
        return np.asarray(result.get_output_ret_joint(), dtype=np.float64)

    def _execute_sub_step(
        self,
        action: np.ndarray,
        raw_action: np.ndarray,
        timing: dict,
        sub_step: int,
        gripper_action: float,
    ):
        """Advance and send a stateful Cartesian reference at a fixed rate."""
        if self.impedance_mode != "cartesian":
            return super()._execute_sub_step(
                action, raw_action, timing, sub_step, gripper_action
            )

        if sub_step > 0:
            part = "first" if sub_step == 1 else "second"
            print(f"[10Hz->20Hz][standalone] Sub-step {sub_step}/2: {part} half")

        substep_start = time.monotonic()
        self._update_currpos()
        timing["1_update_currpos"] = time.time()
        current_xyzabc = self.currpos_xyzabc.copy()

        pos_delta_mm = action[:3] * self.action_scale[0]
        target_xyzabc = current_xyzabc.copy()
        target_xyzabc[:3] += pos_delta_mm

        action_rot_rad = np.zeros(3, dtype=np.float64)
        if not getattr(self.config, "FIXED_ORIENTATION", False):
            action_rot_rad[2] = action[3] * self.action_scale[1]
        target_xyzabc[3:] += np.rad2deg(action_rot_rad)

        # Match MarvinEnv: lock A/B and allow only C to follow the action.
        target_xyzabc[3] = self._RESET_POSE[3]
        target_xyzabc[4] = self._RESET_POSE[4]
        target_xyzabc = self.clip_safety_box(target_xyzabc)
        self.nextpos = np.concatenate(
            [
                target_xyzabc[:3] / 1000.0,
                Rotation.from_euler(
                    "xyz", np.deg2rad(target_xyzabc[3:])
                ).as_quat(),
            ]
        )

        timing["2a_recover"] = time.time()
        delta_dist_mm = np.linalg.norm(pos_delta_mm)
        delta_rot_rad = np.linalg.norm(action_rot_rad)
        timing["2b_stream"] = time.time()

        if not hasattr(self, "_stream_ref_xyzabc"):
            self._reset_stream_reference()

        sample_count = max(1, round(CARTESIAN_SEND_HZ * CARTESIAN_SUBSTEP_SECONDS))
        period = 1.0 / CARTESIAN_SEND_HZ
        send_start = time.monotonic()
        missed_deadlines = 0
        ik_failures = 0

        for index in range(sample_count):
            self._stream_ref_xyzabc[:3], self._stream_linear_velocity = (
                self._step_vector_reference(
                    self._stream_ref_xyzabc[:3],
                    self._stream_linear_velocity,
                    target_xyzabc[:3],
                    STREAM_LINEAR_VEL_MM_S,
                    STREAM_LINEAR_ACC_MM_S2,
                    period,
                )
            )
            self._stream_ref_xyzabc[5], self._stream_yaw_velocity = (
                self._step_scalar_reference(
                    self._stream_ref_xyzabc[5],
                    self._stream_yaw_velocity,
                    target_xyzabc[5],
                    STREAM_YAW_VEL_DEG_S,
                    STREAM_YAW_ACC_DEG_S2,
                    period,
                )
            )
            self._stream_ref_xyzabc[3:5] = self._RESET_POSE[3:5]

            joint_reference = self._stream_ik(self._stream_ref_xyzabc)
            if joint_reference is None:
                ik_failures += 1
                self._stream_linear_velocity.fill(0.0)
                self._stream_yaw_velocity = 0.0
            else:
                self._stream_ref_joints = joint_reference
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=joint_reference)
                self.robot.send_cmd()

            if index + 1 < sample_count:
                deadline = send_start + (index + 1) * period
                sleep_seconds = deadline - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                else:
                    missed_deadlines += 1

        send_elapsed_ms = (time.monotonic() - send_start) * 1000.0
        print(
            f"[standalone][Cartesian stream] target_delta="
            f"{delta_dist_mm:.3f} mm / {np.rad2deg(delta_rot_rad):.3f} deg, "
            f"sent={sample_count} at {CARTESIAN_SEND_HZ} Hz in "
            f"{send_elapsed_ms:.1f} ms; ik_failures={ik_failures}, "
            f"missed_deadlines={missed_deadlines}"
        )
        timing.setdefault("stream_substeps", []).append(
            {
                "substep": sub_step,
                "send_frequency_hz": CARTESIAN_SEND_HZ,
                "sent_point_count": sample_count,
                "playback_duration_seconds": CARTESIAN_SUBSTEP_SECONDS,
                "linear_velocity_limit_mm_s": STREAM_LINEAR_VEL_MM_S,
                "linear_acceleration_limit_mm_s2": STREAM_LINEAR_ACC_MM_S2,
                "yaw_velocity_limit_deg_s": STREAM_YAW_VEL_DEG_S,
                "yaw_acceleration_limit_deg_s2": STREAM_YAW_ACC_DEG_S2,
                "send_elapsed_seconds": send_elapsed_ms / 1000.0,
                "ik_failures": ik_failures,
                "missed_deadlines": missed_deadlines,
            }
        )
        timing["2c_send_joints"] = time.time()
        timing["2d_mode_check"] = time.time()

        self._send_gripper_command(gripper_action)
        timing["3_gripper"] = time.time()

        remaining = CARTESIAN_SUBSTEP_SECONDS - (
            time.monotonic() - substep_start
        )
        if remaining > 0:
            time.sleep(remaining)
        else:
            print(
                f"[standalone][Cartesian] substep overrun: "
                f"{-remaining * 1000.0:.1f} ms"
            )
        timing["4_sleep"] = time.time()

        self.last_joints = self.curr_joints.copy()
        if sub_step == 0 or sub_step == 2:
            self.curr_path_length += 1

    def execute_base_action_without_observation(
        self, base_action: np.ndarray, transformed_unclipped: np.ndarray
    ):
        """Mirror MarvinEnv.step() control execution, omitting obs and reward."""
        action = np.asarray(base_action, dtype=np.float32).copy()
        position_limit = float(getattr(self.config, "POSITION_ACTION_LIMIT", 1.0))
        max_translation_norm = np.sqrt(3.0) * position_limit
        translation_norm = np.linalg.norm(action[:3])
        if translation_norm > max_translation_norm:
            action[:3] *= max_translation_norm / translation_norm
        action[3:] = np.clip(
            action[3:], self.action_space.low[3:], self.action_space.high[3:]
        )
        self.last_action = action.copy()
        timing = {"0_start": time.time()}

        if (
            self.impedance_mode == "cartesian"
            and getattr(self.config, "USE_STREAMING_CARTESIAN", True)
        ):
            self._execute_cartesian_stream_step(
                action, transformed_unclipped, timing, gripper_action=action[4]
            )
            substeps = [action.copy()]
        elif self.hz == 10:
            half_action = action.copy()
            half_action[:4] /= 2.0
            self._execute_sub_step(
                half_action,
                transformed_unclipped,
                timing,
                sub_step=1,
                gripper_action=action[4],
            )
            self._execute_sub_step(
                half_action,
                transformed_unclipped,
                timing,
                sub_step=2,
                gripper_action=action[4],
            )
            substeps = [half_action.copy(), half_action.copy()]
        else:
            self._execute_sub_step(
                action,
                transformed_unclipped,
                timing,
                sub_step=0,
                gripper_action=action[4],
            )
            substeps = [action.copy()]

        self._update_currpos()
        self.last_replay_timing = timing
        return action, substeps


def natural_key(path: str):
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", path)]


def choose_pickle_file(sample_path: str, rng: random.Random) -> str:
    path = os.path.abspath(os.path.expanduser(sample_path))
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Sample path does not exist: {path}")

    files = sorted(
        glob.glob(os.path.join(path, "transitions_*.pkl")), key=natural_key
    )
    if not files:
        raise FileNotFoundError(f"No transitions_*.pkl files found in {path}")
    return rng.choice(files)


def split_episodes(transitions: list[dict]) -> list[list[dict]]:
    episodes: list[list[dict]] = []
    current: list[dict] = []
    for transition in transitions:
        if "actions" not in transition:
            continue
        current.append(transition)
        if bool(transition.get("dones", False)):
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return [episode for episode in episodes if episode]


def load_sampled_episode(
    sample_path: str,
    rng: random.Random,
    episode_index: int | None,
) -> tuple[str, int, np.ndarray, bool]:
    selected_file = choose_pickle_file(sample_path, rng)
    print(f"[sample] Loading: {selected_file}")
    with open(selected_file, "rb") as file:
        transitions = pickle.load(file)
    episodes = split_episodes(transitions)
    if not episodes:
        raise ValueError(f"No action trajectory found in {selected_file}")

    if episode_index is None:
        selected_index = rng.randrange(len(episodes))
    else:
        if not 0 <= episode_index < len(episodes):
            raise IndexError(
                f"episode-index={episode_index}, but file has {len(episodes)} trajectories"
            )
        selected_index = episode_index

    episode = episodes[selected_index]
    actions = np.asarray([transition["actions"] for transition in episode], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 5:
        raise ValueError(f"Expected actions with shape [T, 5], got {actions.shape}")
    complete = bool(episode[-1].get("dones", False))
    return selected_file, selected_index, actions, complete


def ee_action_to_base(action: np.ndarray, tcp_quat_pose: np.ndarray) -> np.ndarray:
    """Mirror RelativeFrame's current 5-D Marvin action mapping."""
    action = np.asarray(action, dtype=np.float32)
    rotation = Rotation.from_quat(tcp_quat_pose[3:]).as_matrix()

    action_6d = np.zeros(6, dtype=np.float64)
    action_6d[:3] = action[:3]
    # The training wrapper maps Marvin's policy drz through EE-y, then retains
    # base-frame rz.  Preserve that exact convention for comparable behavior.
    action_6d[4] = action[3]
    transformed_6d = np.zeros(6, dtype=np.float64)
    transformed_6d[:3] = rotation @ action_6d[:3]
    transformed_6d[3:] = rotation @ action_6d[3:]

    result = action.copy()
    result[:3] = transformed_6d[:3]
    result[3] = transformed_6d[5]
    return result


def ndarray_list(value: np.ndarray) -> list[float]:
    return np.asarray(value).astype(float).tolist()


def describe_actions(actions: np.ndarray):
    print(f"[sample] Steps: {len(actions)}")
    for index, name in enumerate(ACTION_NAMES):
        values = actions[:, index]
        print(
            f"  {name:7s}: min={values.min(): .4f}, max={values.max(): .4f}, "
            f"mean={values.mean(): .4f}, std={values.std(): .4f}"
        )


def config_summary(config: StandaloneControlConfig, ema_alpha: float) -> dict:
    return {
        "robot_ip": config.ROBOT_IP,
        "arm": config.ARM,
        "hz": config.HZ,
        "action_scale": ndarray_list(config.ACTION_SCALE),
        "ema_alpha": ema_alpha,
        "cartesian_send_hz": CARTESIAN_SEND_HZ,
        "cartesian_substep_seconds": CARTESIAN_SUBSTEP_SECONDS,
        "stream_linear_velocity_mm_s": STREAM_LINEAR_VEL_MM_S,
        "stream_linear_acceleration_mm_s2": STREAM_LINEAR_ACC_MM_S2,
        "stream_yaw_velocity_deg_s": STREAM_YAW_VEL_DEG_S,
        "stream_yaw_acceleration_deg_s2": STREAM_YAW_ACC_DEG_S2,
        "fixed_orientation": config.FIXED_ORIENTATION,
        "impedance_mode": config.IMPEDANCE_MODE,
        "velocity_ratio": config.VEL_RATIO,
        "acceleration_ratio": config.ACC_RATIO,
        "compliance_k": ndarray_list(config.COMPLIANCE_PARAM["K"]),
        "compliance_d": ndarray_list(config.COMPLIANCE_PARAM["D"]),
        "precision_k": ndarray_list(config.PRECISION_PARAM["K"]),
        "precision_d": ndarray_list(config.PRECISION_PARAM["D"]),
        "joint_compliance_k": ndarray_list(config.JOINT_COMPLIANCE_PARAM["K"]),
        "joint_compliance_d": ndarray_list(config.JOINT_COMPLIANCE_PARAM["D"]),
        "tool_kine": ndarray_list(config.TOOL_KINE_PARAMS),
        "tool_dynamic": ndarray_list(config.TOOL_DYN_PARAMS),
        "gripper_sleep": config.GRIPPER_SLEEP,
        "safety_low": ndarray_list(config.ABS_POSE_LIMIT_LOW),
        "safety_high": ndarray_list(config.ABS_POSE_LIMIT_HIGH),
        "reset_pose": ndarray_list(config.RESET_POSE),
    }


def default_output_path() -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(PROJECT_ROOT / "utils/action_replay_logs" / f"marvin_replay_{timestamp}.json")


def save_log(path: str, payload: dict):
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[log] Saved trajectory: {output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a sampled Marvin action trajectory without observations."
    )
    parser.add_argument("--sample-path", default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="EMA current-action weight; default uses DEFAULT_EMA_ALPHA in this file.",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="0 executes all steps.")
    parser.add_argument("--dry-run", action="store_true", help="Load/filter/log only; do not connect to robot.")
    parser.add_argument("--yes", action="store_true", help="Skip the real-robot confirmation prompt.")
    parser.add_argument("--return-to-reset", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)
    config = StandaloneControlConfig()
    ema_alpha = (
        args.ema_alpha
        if args.ema_alpha is not None
        else DEFAULT_EMA_ALPHA
    )

    source_file, episode_index, actions, complete = load_sampled_episode(
        args.sample_path, rng, args.episode_index
    )
    start = max(0, args.start_step)
    stop = len(actions) if args.max_steps <= 0 else min(len(actions), start + args.max_steps)
    actions = actions[start:stop]
    if len(actions) == 0:
        raise ValueError("Selected trajectory slice is empty")

    print(f"[sample] Random seed: {seed}")
    print(f"[sample] Episode/segment: {episode_index} (complete={complete})")
    print(f"[sample] Slice: [{start}:{stop}]")
    describe_actions(actions)
    print("\n[control configuration]")
    print(json.dumps(config_summary(config, ema_alpha), ensure_ascii=False, indent=2))

    payload = {
        "source_file": source_file,
        "episode_index": episode_index,
        "source_complete_episode": complete,
        "slice": [start, stop],
        "seed": seed,
        "sample_actions_are_native": True,
        "config": config_summary(config, ema_alpha),
        "records": [],
    }
    output_path = args.output or default_output_path()
    action_filter = ContinuousActionEMA(ema_alpha)
    action_filter.reset()

    if args.dry_run:
        for step, raw_action in enumerate(actions):
            filtered = action_filter(raw_action)
            if (
                config.IMPEDANCE_MODE == "cartesian"
                and getattr(config, "USE_STREAMING_CARTESIAN", True)
            ):
                substeps = [filtered.tolist()]
            elif config.HZ == 10:
                substeps = [np.r_[filtered[:4] / 2.0, filtered[4]].tolist()] * 2
            else:
                substeps = [filtered.tolist()]
            payload["records"].append(
                {
                    "step": step,
                    "raw_ee_action": ndarray_list(raw_action),
                    "filtered_ee_action": ndarray_list(filtered),
                    "base_action": None,
                    "substeps": substeps,
                }
            )
        save_log(output_path, payload)
        print("[dry-run] Robot was not connected.")
        return 0

    print("\n" + "=" * 72)
    print("WARNING: This will connect to and move the real Marvin robot.")
    print("Verify the workspace, USB setup, and physical emergency stop first.")
    print("Ctrl+C triggers the environment emergency-stop handler.")
    print("=" * 72)
    if not args.yes:
        input("Press Enter to connect and continue (Ctrl+C to cancel): ")

    env: NoObservationMarvinUSBEnv | None = None
    try:
        env = NoObservationMarvinUSBEnv(
            hz=config.HZ,
            fake_env=False,
            save_video=False,
            config=config,
        )
        env.reset_without_observation()
        if not args.yes:
            input(
                "Robot is at reset. Prepare the USB/workspace, then press Enter "
                "to execute (Ctrl+C to cancel): "
            )

        for step, raw_action in enumerate(actions):
            filtered = action_filter(raw_action)
            env._update_currpos()
            pose_before = env.currpos_xyzabc.copy()
            transformed_unclipped = ee_action_to_base(filtered, env.currpos_quat)

            started = time.time()
            executed_base, substeps = env.execute_base_action_without_observation(
                transformed_unclipped, transformed_unclipped
            )
            elapsed = time.time() - started
            pose_after = env.currpos_xyzabc.copy()

            record = {
                "step": step,
                "raw_ee_action": ndarray_list(raw_action),
                "filtered_ee_action": ndarray_list(filtered),
                "base_action_unclipped": ndarray_list(transformed_unclipped),
                "base_action": ndarray_list(executed_base),
                "substeps": [ndarray_list(value) for value in substeps],
                "stream_execution": getattr(env, "last_replay_timing", {}).get(
                    "stream_substeps", []
                ),
                "tcp_before_xyzabc": ndarray_list(pose_before),
                "tcp_after_xyzabc": ndarray_list(pose_after),
                "elapsed_seconds": elapsed,
            }
            payload["records"].append(record)
            print(
                f"[replay {step + 1:03d}/{len(actions):03d}] "
                f"raw={np.round(raw_action, 3)} "
                f"ema={np.round(filtered, 3)} "
                f"tcp={np.round(pose_after[:3], 1)} mm"
            )

        if args.return_to_reset:
            print("[standalone] Returning to reset pose...")
            env.reset_without_observation()

        save_log(output_path, payload)
        print("[standalone] Replay completed successfully.")
        return 0
    except Exception as error:
        payload["error"] = repr(error)
        payload["completed_steps"] = len(payload["records"])
        save_log(output_path, payload)
        print(f"[standalone] Replay failed after {len(payload['records'])} steps: {error}")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
