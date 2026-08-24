"""
Marvin USB Insertion任务特定环境和Wrapper

实现Marvin机器人的USB插拔任务逻辑
参考: franka的usb_pickup_insertion/wrapper.py
"""
import json
import numpy as np
import time
import gymnasium as gym
from pathlib import Path
from typing import OrderedDict

from marvin_env.envs.marvin_env import MarvinEnv
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.audio_utils import get_audio_notifier


class MarvinUSBEnv(MarvinEnv):
    """
    Marvin USB插拔任务环境

    继承自MarvinEnv，实现USB任务特定的reset逻辑：
    1. 移动到USB上方并打开夹爪
    2. 下降夹取并后退抽出
    3. 移动到随机投放区并释放
    4. 回到标定关节起始位
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 初始化音频通知器
        self.audio_notifier = get_audio_notifier(
            device="plughw:3,0",
            enabled=True  # 设置为 False 可以禁用音频
        )

    def init_cameras(self, name_serial_dict=None):
        """
        初始化相机（支持相机共享）

        side_policy和side_classifier可以共享同一个相机
        """
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        # 检查是否有相机共享同一个序列号
        serial_to_name = {}
        for cam_name, kwargs in name_serial_dict.items():
            serial = kwargs.get("serial_number")
            if serial in serial_to_name:
                # 共享已有相机
                existing_name = serial_to_name[serial]
                print(f"[相机共享] {cam_name} 共享 {existing_name} 的相机 (序列号: {serial})")
                self.cap[cam_name] = self.cap[existing_name]
            else:
                # 创建新相机
                print(f"[相机初始化] {cam_name} (序列号: {serial})")
                cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
                self.cap[cam_name] = cap
                serial_to_name[serial] = cam_name

    def reset(self, **kwargs):
        """
        重置环境。

        复位选择:

        RESET_BY_EPISODE_RESULT = True:
            上一回合成功执行自动复位；失败、超时或首次启动执行手动复位。

        RESET_BY_EPISODE_RESULT = False:
            继续由 AUTO_RESET_USB 固定选择以下两种模式。

        AUTO_RESET_USB = True（自动模式）:
            1. 错误恢复并验证阻抗模式
            2. 示教点上方 → 下降夹取 → 后退抽出
            3. 随机投放并释放 → 移动到标定关节起始位

        AUTO_RESET_USB = False（手动模式）:
            1. 错误恢复
            2. 机械臂归位到标定关节起始位 + 打开夹爪
            3. 等待 MANUAL_RESET_TIMEOUT 秒供人放置USB
            4. 柔顺模式 → 开始episode
        """
        # 检查是否需要周期性关节重置
        if self.curr_path_length != 0 and self.joint_reset_cycle != 0:
            if self.curr_path_length % self.joint_reset_cycle == 0:
                print("[MarvinUSBEnv] 执行周期性关节重置")
                self._recover()
                self._ensure_compliance_mode()
                self.go_to_reset(joint_reset=True)
                self._recover()
                self._ensure_compliance_mode()
                self.curr_path_length = 0
                return self._get_obs(), {"succeed": False}

        print("\n[MarvinUSBEnv] 执行重置...")

        # ==================== 错误恢复 ====================
        self._recover()
        self._ensure_compliance_mode()
        self._update_currpos()
        time.sleep(0.1)

        # ==================== USB复位 (自动/手动) ====================
        self._reset_usb_for_last_episode()

        # ==================== 重置Episode ====================
        self.curr_path_length = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        self._update_currpos()
        self._reset_stream_reference()

        print("[MarvinUSBEnv] 重置完成，开始episode\n")

        obs = self._get_obs()
        return obs, {"succeed": False}

    def set_last_episode_result(self, succeeded: bool):
        """接收外层奖励 wrapper 判定的上一回合结果。"""
        self._last_episode_succeeded = bool(succeeded)

    def _reset_usb_for_last_episode(self):
        """按上一回合结果选择复位；结果只消费一次。"""
        if getattr(self.config, 'RESET_BY_EPISODE_RESULT', False):
            succeeded = getattr(self, '_last_episode_succeeded', None)
            self._last_episode_succeeded = None
            if succeeded is True:
                print("[MarvinUSBEnv] 上一回合成功，执行自动复位")
                self._auto_reset_usb()
            else:
                reason = "首次启动/无结果" if succeeded is None else "上一回合失败"
                print(f"[MarvinUSBEnv] {reason}，执行手动复位")
                self._manual_reset_usb()
            return

        if getattr(self.config, 'AUTO_RESET_USB', False):
            self._auto_reset_usb()
        else:
            self._manual_reset_usb()

    def _movla_to_xyzabc(
        self,
        target_xyzabc: np.ndarray,
        auto_segment: bool = True,
        vel: int = 100,
        acc: int = 100,
    ):
        """
        执行笛卡尔空间线性移动

        Args:
            target_xyzabc: 目标位姿 [X,Y,Z,A,B,C] (mm, deg)
            auto_segment: 是否自动分段（大距离时插入中间点）
        """
        self._update_currpos()
        current_xyzabc = self.currpos_xyzabc.copy()

        # 打印当前模式
        sub = self.robot.subscribe(self.dcss)
        cur_state = sub['states'][self.arm_idx]['cur_state']
        imp_type = sub['inputs'][self.arm_idx]['imp_type']

        # 计算距离
        pos_delta = target_xyzabc[:3] - current_xyzabc[:3]
        rot_delta = target_xyzabc[3:] - current_xyzabc[3:]
        pos_distance = np.linalg.norm(pos_delta)  # mm
        rot_distance = np.linalg.norm(rot_delta)  # deg

        print(f"[_movla_to_xyzabc] 当前模式: state={cur_state}, imp_type={imp_type}")
        print(f"[_movla_to_xyzabc] 距离: 位置={pos_distance:.1f}mm, 姿态={rot_distance:.1f}deg")

        # 🎯 智能分段：根据距离决定分段数
        if auto_segment:
            if pos_distance > 150 or rot_distance > 25:
                num_segments = 3
                print(f"[_movla_to_xyzabc] ⚠️ 距离过大，分{num_segments}段执行")
            elif pos_distance > 80 or rot_distance > 15:
                num_segments = 2
                print(f"[_movla_to_xyzabc] ⚠️ 距离较大，分{num_segments}段执行")
            else:
                num_segments = 1
        else:
            num_segments = 1

        # 执行分段移动
        for seg in range(num_segments):
            # 计算本段的目标位姿（线性插值）
            ratio = (seg + 1) / num_segments
            waypoint_xyzabc = current_xyzabc + (target_xyzabc - current_xyzabc) * ratio

            # 获取当前关节角度
            self._update_currpos()
            current_joints = self.curr_joints.tolist()
            current_pose = self.currpos_xyzabc.copy()

            if num_segments > 1:
                print(f"[_movla_to_xyzabc] 第{seg+1}/{num_segments}段: {current_pose[:3]} → {waypoint_xyzabc[:3]}")

            # 规划轨迹
            points, _ = self.kk.movLA(
                start_xyzabc=current_pose.tolist(),
                end_xyzabc=waypoint_xyzabc.tolist(),
                ref_joints=current_joints,
                vel=vel,
                acc=acc,
                freq_hz=100
            )

            if not points or len(points) == 0:
                print(f"[_movla_to_xyzabc] 警告: 第{seg+1}段轨迹规划失败")
                continue

            # 执行轨迹
            if num_segments == 1:
                print(f"[_movla_to_xyzabc] 开始执行 {len(points)} 个轨迹点...")

            for i, pt in enumerate(points):
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                self.robot.send_cmd()
                time.sleep(0.002)  # 2ms per point = 500Hz

                # 每 100 点打印一次进度
                if num_segments == 1 and ((i + 1) % 100 == 0 or i == len(points) - 1):
                    print(f"  [{i+1}/{len(points)}] 执行中...")

            if num_segments > 1:
                print(f"[_movla_to_xyzabc] 第{seg+1}/{num_segments}段完成")
                time.sleep(0.1)  # 段间停顿，确保到位

        print(f"[_movla_to_xyzabc] 轨迹执行完成")
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """在当前阻抗模式下回到标定关节角。"""
        sub = self.robot.subscribe(self.dcss)
        cur_state = sub['states'][self.arm_idx]['cur_state']
        imp_type = sub['inputs'][self.arm_idx]['imp_type']
        print(f"[go_to_reset] 当前模式: state={cur_state}, imp_type={imp_type}")

        expected_imp_type = 1 if self.impedance_mode == 'joint' else 2
        if cur_state != 3 or imp_type != expected_imp_type:
            raise RuntimeError(
                "关节复位前阻抗模式异常: "
                f"state={cur_state}, imp_type={imp_type}, "
                f"期望 state=3, imp_type={expected_imp_type}"
            )
        if self.randomreset:
            raise RuntimeError("关节起始位已启用，RANDOM_RESET 必须为 False")

        # 🔧 在移动之前先打开夹爪
        if self.config.USE_GRIPPER:
            print("[go_to_reset] 打开夹爪...")
            self._send_gripper_command(1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        self._move_to_reset_joints()
        self._update_currpos()
        self._reset_stream_reference()
        time.sleep(0.5)

    def _load_reset_joints(self):
        path = Path(self.config.RESET_JOINTS_PATH)
        if not path.is_file():
            raise RuntimeError(
                f"未找到起始关节标定: {path}\n"
                "请先运行 calibrate_reset_joints.py"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"不支持的关节标定版本: {payload.get('schema_version')}")
        if payload.get("arm") != self.arm:
            raise ValueError(
                f"标定文件属于 {payload.get('arm')} 臂，当前是 {self.arm} 臂"
            )
        calibrated_pose = np.asarray(
            payload.get("reset_pose_xyzabc"), dtype=np.float64
        )
        if calibrated_pose.shape != (6,) or not np.allclose(
            calibrated_pose, self.config.RESET_POSE, atol=1e-6
        ):
            raise ValueError("RESET_POSE 已改变，请重新运行关节起始位标定")
        joints = np.asarray(payload.get("joints_deg"), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"标定关节角无效: {joints}")
        return joints

    def _move_to_reset_joints(self):
        """阻抗模式下使用五次曲线平滑回到唯一关节起始位。"""
        target = self._load_reset_joints()
        self._update_currpos()
        start = np.asarray(self.curr_joints, dtype=np.float64).copy()
        max_delta = float(np.max(np.abs(target - start)))
        max_speed = float(self.config.RESET_JOINT_MAX_SPEED_DEG_S)
        hz = int(self.config.RESET_JOINT_COMMAND_HZ)
        if max_speed <= 0.0 or hz <= 0:
            raise ValueError("关节复位速度/频率配置无效")

        duration = max(
            float(self.config.RESET_JOINT_MOVE_MIN_DURATION), max_delta / max_speed
        )
        steps = max(2, int(np.ceil(duration * hz)))
        print(
            f"[go_to_reset] 关节起始位: 最大变化={max_delta:.2f}°, "
            f"时长={duration:.2f}s, 指令点={steps}"
        )
        for phase in np.linspace(0.0, 1.0, steps + 1)[1:]:
            blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(
                arm=self.arm, joints=(start + blend * (target - start)).tolist()
            )
            self.robot.send_cmd()
            time.sleep(1.0 / hz)

        deadline = time.monotonic() + float(self.config.RESET_JOINT_SETTLE_TIMEOUT)
        tolerance = float(self.config.RESET_JOINT_TOLERANCE_DEG)
        while True:
            self._update_currpos()
            error = float(np.max(np.abs(target - self.curr_joints)))
            if error <= tolerance:
                print(f"[go_to_reset] 关节起始位到位，最大误差={error:.3f}°")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"关节起始位未到位: 最大误差={error:.3f}°, "
                    f"容差={tolerance:.3f}°"
                )
            time.sleep(0.05)

    def _open_gripper_and_wait(self, context):
        """自动复位专用张爪联锁：确认实测开度后才返回。"""
        if not self.config.USE_GRIPPER:
            return
        if self.gripper is None:
            raise RuntimeError(f"{context}: 夹爪未初始化，停止自动复位")

        threshold = float(self.config.AUTO_RESET_GRIPPER_OPEN_THRESHOLD)
        timeout = float(self.config.AUTO_RESET_GRIPPER_OPEN_TIMEOUT)
        if not 0.0 <= threshold <= 1.0 or timeout < 0.0:
            raise ValueError("自动复位夹爪打开阈值/超时配置无效")

        last_position = None
        for attempt in range(1, 3):
            if not self.gripper.open(blocking=True):
                print(f"[MarvinUSBEnv] {context}: 张爪指令失败，重试 {attempt}/2")
                continue

            deadline = time.monotonic() + timeout
            while True:
                last_position = float(self.gripper.get_normalized_position())
                self.curr_gripper_pos = last_position
                if last_position >= threshold:
                    print(
                        f"[MarvinUSBEnv] {context}: 夹爪已打开 "
                        f"({last_position:.3f} >= {threshold:.3f})"
                    )
                    return
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            print(
                f"[MarvinUSBEnv] {context}: 开度不足 "
                f"({last_position:.3f} < {threshold:.3f})，重试 {attempt}/2"
            )

        measured = "无反馈" if last_position is None else f"{last_position:.3f}"
        raise RuntimeError(
            f"{context}: 夹爪未确认打开 (实测={measured}, 阈值={threshold:.3f})，"
            "停止自动复位，禁止机械臂抬升"
        )

    def _release_gripper_best_effort(self, context):
        """投放阶段强制发送张爪指令，不以反馈开度为前提。

        外力卡住时允许指令返回失败，仍然继续复位，并在回到起始位后重试一次。
        """
        if not self.config.USE_GRIPPER or self.gripper is None:
            return
        try:
            succeeded = self.gripper.open(blocking=False)
            if not succeeded:
                print(f"[MarvinUSBEnv] {context}: 张爪指令未成功，继续复位")
        except Exception as exc:
            print(f"[MarvinUSBEnv] {context}: 张爪异常 ({exc})，继续复位")
        time.sleep(self.config.AUTO_RESET_RELEASE_SETTLE_TIME)

    def _auto_reset_usb(self):
        """按示教关键点抽出并随机投放 USB，最后回到标定关节起始位。"""
        sub = self.robot.subscribe(self.dcss)
        cur_state = sub['states'][self.arm_idx]['cur_state']
        imp_type = sub['inputs'][self.arm_idx]['imp_type']
        expected_imp_type = 1 if self.impedance_mode == 'joint' else 2
        if cur_state != 3 or imp_type != expected_imp_type:
            raise RuntimeError(
                "自动复位前阻抗模式异常: "
                f"state={cur_state}, imp_type={imp_type}, "
                f"期望 state=3, imp_type={expected_imp_type}"
            )

        waypoints = self.config.AUTO_RESET_WAYPOINTS
        required = ("usb_above", "usb_grasp", "extract_retreat", "drop_center")
        if set(waypoints) != set(required):
            raise ValueError(f"AUTO_RESET_WAYPOINTS 必须且只能包含: {required}")

        def move_to(name, target):
            target = np.asarray(target, dtype=np.float64)
            if target.shape != (6,) or not np.all(np.isfinite(target)):
                raise ValueError(f"自动复位点 {name} 无效: {target}")
            if np.any(target < self.config.ABS_POSE_LIMIT_LOW) or np.any(
                target > self.config.ABS_POSE_LIMIT_HIGH
            ):
                raise ValueError(f"自动复位点 {name} 超出安全边界: {target}")
            print(f"[MarvinUSBEnv] 自动复位 -> {name}: {target}")
            self._movla_to_xyzabc(
                target,
                vel=self.config.RESET_MOVLA_VEL,
                acc=self.config.RESET_MOVLA_ACC,
            )
            time.sleep(0.3)

        # 1. 到达光模块上方后打开夹爪。
        move_to("usb_above", waypoints["usb_above"])
        self._open_gripper_and_wait("抓取前张爪")

        # 2. 下落到两侧并夹紧。
        move_to("usb_grasp", waypoints["usb_grasp"])
        if self.config.USE_GRIPPER:
            self._send_gripper_command(-1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        # 3. 后退抽出光模块。
        move_to("extract_retreat", waypoints["extract_retreat"])

        # 4. 在投放中心的 XYZ 半区间内随机选择目标，姿态保持示教值。
        drop_target = np.asarray(waypoints["drop_center"], dtype=np.float64).copy()
        half_range = np.asarray(
            self.config.AUTO_RESET_DROP_RANDOM_HALF_RANGE_MM, dtype=np.float64
        )
        if half_range.shape != (3,) or np.any(half_range < 0):
            raise ValueError(
                "AUTO_RESET_DROP_RANDOM_HALF_RANGE_MM 必须是三个非负数"
            )
        drop_target[:3] += np.random.uniform(-half_range, half_range)
        move_to("drop_random", drop_target)
        self._release_gripper_best_effort("投放后张爪")

        # 5. 两套复位逻辑唯一共用的关节起始位。
        self._move_to_reset_joints()
        self._release_gripper_best_effort("回位后重试张爪")

    def _manual_reset_usb(self):
        """手动复位: 机械臂归位 -> 等待人工放置USB"""
        print("[MarvinUSBEnv] 手动复位模式: 机械臂归位...")

        # 1. 机械臂回到标定关节起始位
        self.go_to_reset(joint_reset=False)

        # 2. 打开夹爪 (如果USB已抓在手中则释放)
        if self.config.USE_GRIPPER:
            print("[MarvinUSBEnv] 打开夹爪...")
            self._send_gripper_command(1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        # 给人工留出放置 USB 的时间，避免 reset 后立即开始下一回合。
        settle_time = float(getattr(self.config, "MANUAL_RESET_SETTLE_TIME", 2.0))
        if settle_time > 0.0:
            print(f"[MarvinUSBEnv] 等待人工放置 USB ({settle_time:.1f}s)...")
            time.sleep(settle_time)

        # 3. 等待人工放置USB
        timeout = getattr(self.config, 'MANUAL_RESET_TIMEOUT', 10.0)
        # print(f"\n{'='*60}")
        # print(f"[MarvinUSBEnv] ⏳ 等待人工放置USB ({timeout:.0f}秒)...")
        # print(f"   机械臂已归位到 RESET_POSE")
        # print(f"   请在 {timeout:.0f} 秒内完成 USB 放置")
        # print(f"{'='*60}")

        # for remaining in range(int(timeout), 0, -1):
        #     print(f"\r   剩余 {remaining} 秒...", end="", flush=True)
        #     time.sleep(1)
        # print("\r   时间到！开始新episode.      ")

        # print("[MarvinUSBEnv] 手动复位完成")


class GripperPenaltyWrapper(gym.Wrapper):
    """
    夹爪动作惩罚Wrapper

    对频繁的夹爪开合动作施加惩罚，鼓励智能体保持夹爪状态
    """

    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (5,), "需要5维动作空间"
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # GripperPenaltyWrapper 在 SERLObsWrapper 之前,
        # obs["state"] 是 dict 结构, 取 gripper_pose
        self.last_gripper_pos = obs["state"]["gripper_pose"]
        return obs, info

    def step(self, action):
        """在step中计算夹爪惩罚"""
        observation, reward, terminated, truncated, info = self.env.step(action)

        # 如果有人类干预，使用干预后的动作
        if "intervene_action" in info:
            action = info["intervene_action"]

        # 计算夹爪惩罚
        # gripper_pose: 0=closed, 1=open
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.95) or \
           (action[-1] > 0.5 and self.last_gripper_pos < 0.05):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        # 更新夹爪位置
        self.last_gripper_pos = observation["state"]["gripper_pose"]

        return observation, reward, terminated, truncated, info
