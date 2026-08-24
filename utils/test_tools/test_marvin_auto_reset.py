import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import gymnasium as gym

from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig
from examples.experiments.marvin_usb_insertion.wrapper import MarvinUSBEnv
from franka_env.envs.wrappers import MultiCameraBinaryRewardClassifierWrapper


class _OutcomeSink(gym.Env):
    def __init__(self):
        self.last_result = None

    def step(self, action):
        return {}, 0.0, False, False, {}

    def reset(self, **kwargs):
        return {}, {}

    def set_last_episode_result(self, succeeded):
        self.last_result = succeeded


class MarvinAutoResetTest(unittest.TestCase):
    def test_ensure_compliance_mode_retries_after_recovery(self):
        env = object.__new__(MarvinUSBEnv)
        env.impedance_mode = "cartesian"
        env.arm_idx = 0
        env.dcss = object()
        env.robot = Mock()
        env.robot.subscribe.side_effect = [
            {"states": [{"cur_state": 0}], "inputs": [{"imp_type": 2}]},
            {"states": [{"cur_state": 3}], "inputs": [{"imp_type": 2}]},
        ]
        env._switch_to_compliance_mode = Mock()

        env._ensure_compliance_mode()

        env._switch_to_compliance_mode.assert_called_once_with()

    def test_ensure_compliance_mode_stops_if_retry_failed(self):
        env = object.__new__(MarvinUSBEnv)
        env.impedance_mode = "cartesian"
        env.arm_idx = 0
        env.dcss = object()
        env.robot = Mock()
        env.robot.subscribe.return_value = {
            "states": [{"cur_state": 0}],
            "inputs": [{"imp_type": 2}],
        }
        env._switch_to_compliance_mode = Mock()

        with self.assertRaisesRegex(RuntimeError, "禁止继续运动"):
            env._ensure_compliance_mode()

    def test_episode_result_selects_reset_and_is_consumed(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env._auto_reset_usb = Mock()
        env._manual_reset_usb = Mock()

        env.set_last_episode_result(True)
        env._reset_usb_for_last_episode()
        env._auto_reset_usb.assert_called_once_with()
        env._manual_reset_usb.assert_not_called()

        env._auto_reset_usb.reset_mock()
        env._reset_usb_for_last_episode()
        env._auto_reset_usb.assert_not_called()
        env._manual_reset_usb.assert_called_once_with()

    def test_reward_wrapper_passes_terminal_result_to_base_env(self):
        sink = _OutcomeSink()
        wrapper = MultiCameraBinaryRewardClassifierWrapper(
            sink, lambda obs: 1.0, terminal_failure_reward=-0.5
        )
        wrapper.step(None)
        self.assertIs(sink.last_result, True)

        wrapper.reward_classifier_func = lambda obs: -0.5
        wrapper.step(None)
        self.assertIs(sink.last_result, False)

    def test_waypoint_sequence_random_drop_and_shared_reset_pose(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env.impedance_mode = "cartesian"
        env.arm_idx = 0
        env.dcss = object()
        env.robot = Mock()
        env.robot.subscribe.return_value = {
            "states": [{"cur_state": 3}],
            "inputs": [{"imp_type": 2}],
        }
        env.gripper = Mock()
        env.gripper.open.return_value = True
        env.gripper.get_normalized_position.return_value = 1.0
        env.curr_gripper_pos = 0.0
        env._movla_to_xyzabc = Mock()
        env._move_to_reset_joints = Mock()
        env._send_gripper_command = Mock()
        drop_offset = np.array([5.0, -6.0, 0.0])

        with patch(
            "examples.experiments.marvin_usb_insertion.wrapper.time.sleep"
        ), patch(
            "examples.experiments.marvin_usb_insertion.wrapper.np.random.uniform",
            return_value=drop_offset,
        ):
            env._auto_reset_usb()

        targets = [call.args[0] for call in env._movla_to_xyzabc.call_args_list]
        waypoints = env.config.AUTO_RESET_WAYPOINTS
        np.testing.assert_allclose(targets[0], waypoints["usb_above"])
        np.testing.assert_allclose(targets[1], waypoints["usb_grasp"])
        np.testing.assert_allclose(targets[2], waypoints["extract_retreat"])
        np.testing.assert_allclose(
            targets[3][:3], waypoints["drop_center"][:3] + drop_offset
        )
        self.assertEqual(len(targets), 4)
        env._move_to_reset_joints.assert_called_once_with()
        self.assertEqual(
            [call.args[0] for call in env._send_gripper_command.call_args_list],
            [-1.0],
        )
        self.assertEqual(env.gripper.open.call_count, 3)
        self.assertEqual(
            [call.kwargs["blocking"] for call in env.gripper.open.call_args_list],
            [True, False, False],
        )
        self.assertTrue(env.config.AUTO_RESET_USB)

    def test_open_interlock_stops_when_feedback_never_reaches_threshold(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env.config.AUTO_RESET_GRIPPER_OPEN_TIMEOUT = 0.0
        env.gripper = Mock()
        env.gripper.open.return_value = True
        env.gripper.get_normalized_position.return_value = 0.2
        env.curr_gripper_pos = 0.0

        with self.assertRaisesRegex(RuntimeError, "禁止机械臂抬升"):
            env._open_gripper_and_wait("测试张爪")

        self.assertEqual(env.gripper.open.call_count, 2)
        self.assertEqual(env.curr_gripper_pos, 0.2)

    def test_release_open_failure_does_not_abort_reset(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env.gripper = Mock()
        env.gripper.open.return_value = False

        with patch(
            "examples.experiments.marvin_usb_insertion.wrapper.time.sleep"
        ) as sleep:
            env._release_gripper_best_effort("投放释放")

        env.gripper.open.assert_called_once_with(blocking=False)
        sleep.assert_called_once_with(env.config.AUTO_RESET_RELEASE_SETTLE_TIME)

    def test_joint_reset_ends_at_calibrated_joint_target(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env.config.RESET_JOINT_MOVE_MIN_DURATION = 0.0
        env.config.RESET_JOINT_MAX_SPEED_DEG_S = 1000.0
        env.config.RESET_JOINT_COMMAND_HZ = 10
        env.config.RESET_JOINT_SETTLE_TIMEOUT = 0.0
        env.config.RESET_JOINT_TOLERANCE_DEG = 0.01
        env.arm = "A"
        env.robot = Mock()
        target = np.arange(7, dtype=np.float64)
        env.curr_joints = np.zeros(7)
        env._load_reset_joints = Mock(return_value=target)

        def update_feedback():
            if env.robot.send_cmd.call_count:
                env.curr_joints = target.copy()

        env._update_currpos = Mock(side_effect=update_feedback)
        with patch("examples.experiments.marvin_usb_insertion.wrapper.time.sleep"):
            env._move_to_reset_joints()

        final_command = env.robot.set_joint_cmd_pose.call_args.kwargs["joints"]
        np.testing.assert_allclose(final_command, target)

    def test_joint_calibration_must_match_current_reset_pose(self):
        env = object.__new__(MarvinUSBEnv)
        env.config = MarvinUSBEnvConfig()
        env.arm = "A"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reset_joints_A.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "arm": "A",
                        "reset_pose_xyzabc": (env.config.RESET_POSE + 1.0).tolist(),
                        "joints_deg": [0.0] * 7,
                    }
                ),
                encoding="utf-8",
            )
            env.config.RESET_JOINTS_PATH = str(path)
            with self.assertRaisesRegex(ValueError, "RESET_POSE 已改变"):
                env._load_reset_joints()


if __name__ == "__main__":
    unittest.main()
