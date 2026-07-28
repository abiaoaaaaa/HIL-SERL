"""
Marvin USB Insertion任务特定环境和Wrapper

实现Marvin机器人的USB插拔任务逻辑
参考: franka的usb_pickup_insertion/wrapper.py
"""
import numpy as np
import time
import gymnasium as gym
from typing import OrderedDict

from marvin_env.envs.marvin_env import MarvinEnv
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.audio_utils import get_audio_notifier


class MarvinUSBEnv(MarvinEnv):
    """
    Marvin USB插拔任务环境

    继承自MarvinEnv，实现USB任务特定的reset逻辑：
    1. 移动到USB上方
    2. 下降到USB位置
    3. 夹取USB
    4. 抬起到起始位置
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

        支持两种模式（由 config.AUTO_RESET_USB 控制）:

        AUTO_RESET_USB = True（自动模式）:
            1. 错误恢复 → 精密模式 → 打开夹爪
            2. 移动到USB上方 → 下降 → 夹取
            3. 抬起 → 移动到reset位置 → 柔顺模式

        AUTO_RESET_USB = False（手动模式）:
            1. 错误恢复
            2. 机械臂归位到 RESET_POSE + 打开夹爪
            3. 等待 MANUAL_RESET_TIMEOUT 秒供人放置USB
            4. 柔顺模式 → 开始episode
        """
        # 检查是否需要周期性关节重置
        if self.curr_path_length != 0 and self.joint_reset_cycle != 0:
            if self.curr_path_length % self.joint_reset_cycle == 0:
                print("[MarvinUSBEnv] 执行周期性关节重置")
                self._recover()
                self.go_to_reset(joint_reset=True)
                self._recover()
                self.curr_path_length = 0
                return self._get_obs(), {"succeed": False}

        print("\n[MarvinUSBEnv] 执行重置...")

        # ==================== 错误恢复 ====================
        self._recover()
        self._update_currpos()
        time.sleep(0.1)

        # ==================== USB复位 (自动/手动) ====================
        auto_reset = getattr(self.config, 'AUTO_RESET_USB', False)

        if auto_reset:
            self._auto_reset_usb()
        else:
            self._manual_reset_usb()

        # ==================== 重置Episode ====================
        self.curr_path_length = 0
        self._update_currpos()

        print("[MarvinUSBEnv] 重置完成，开始episode\n")

        obs = self._get_obs()
        return obs, {"succeed": False}

    def _movla_to_xyzabc(self, target_xyzabc: np.ndarray, auto_segment: bool = True):
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
                vel=100,
                acc=100,
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
        if joint_reset:
            print("[MarvinUSBEnv] 关节重置暂不支持")

        # 打印当前模式
        sub = self.robot.subscribe(self.dcss)
        cur_state = sub['states'][self.arm_idx]['cur_state']
        imp_type = sub['inputs'][self.arm_idx]['imp_type']
        print(f"[go_to_reset] 当前模式: state={cur_state}, imp_type={imp_type}")

        # 🔧 在移动之前先打开夹爪
        if self.config.USE_GRIPPER:
            print("[go_to_reset] 打开夹爪...")
            self._send_gripper_command(1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        # 🛑 停止所有运动指令，避免模式切换冲突
        print("[go_to_reset] 停止所有运动...")
        self.robot.stopRunPln_joint(arm=self.arm)
        time.sleep(0.5)  # 等待停止生效

        # 切到位置模式 (state=1) - 带重试和验证
        print("[go_to_reset] 切换到位置模式 (state=1)...")

        max_retries = 5
        for attempt in range(max_retries):
            self.robot.clear_set()
            self.robot.set_state(arm=self.arm, state=1)
            self.robot.set_vel_acc(arm=self.arm, velRatio=50, AccRatio=50)
            self.robot.send_cmd()
            time.sleep(0.3)  # 初始等待

            # 验证是否切换成功
            for check in range(5):
                sub = self.robot.subscribe(self.dcss)
                cur_state = sub['states'][self.arm_idx]['cur_state']

                if cur_state == 1:
                    print(f"[go_to_reset] ✓ 切换成功: state={cur_state} (尝试{attempt+1}/{max_retries})")
                    break

                time.sleep(0.2)  # 等待状态稳定

            if cur_state == 1:
                break

            print(f"[go_to_reset] ✗ 切换失败: state={cur_state}, 重试 {attempt+1}/{max_retries}")
            time.sleep(0.5)  # 失败后等久一点再重试

        if cur_state != 1:
            print(f"[go_to_reset] ⚠️ 警告: {max_retries}次尝试后仍未切换到位置模式 (当前state={cur_state})")
            # 不抛异常，继续执行，但打印明显警告

        if self.randomreset:
            reset_pose = self.config.RESET_POSE.copy()
            reset_pose[:2] += np.random.uniform(
                -self.config.RANDOM_XY_RANGE, self.config.RANDOM_XY_RANGE, (2,)
            )
            reset_pose[5] += np.rad2deg(np.random.uniform(
                -self.config.RANDOM_RZ_RANGE, self.config.RANDOM_RZ_RANGE
            ))
            self._movla_to_xyzabc(reset_pose)
        else:
            self._movla_to_xyzabc(self.config.RESET_POSE)

        print("[go_to_reset] 等待到位...")
        time.sleep(0.5)

        print("[go_to_reset] 切回柔顺模式...")
        self._switch_to_compliance_mode()

        # 验证柔顺模式切换
        for check in range(5):
            sub = self.robot.subscribe(self.dcss)
            cur_state = sub['states'][self.arm_idx]['cur_state']
            imp_type = sub['inputs'][self.arm_idx]['imp_type']

            if cur_state == 3 and imp_type == 2:
                print(f"[go_to_reset] ✓ 最终模式验证成功: state={cur_state}, imp_type={imp_type}")
                break

            time.sleep(0.2)
        else:
            print(f"[go_to_reset] ⚠️ 最终模式异常: state={cur_state}, imp_type={imp_type} (期望state=3, imp_type=2)")

    def _auto_reset_usb(self):
        """自动USB抓取流程: 移动到USB上方 -> 下降 -> 夹取 -> 抬起 -> 移动到reset"""
        print("[MarvinUSBEnv] 切换到精密模式...")
        self._switch_to_precision_mode()

        # 1. 打开夹爪
        if self.config.USE_GRIPPER:
            print("[MarvinUSBEnv] 打开夹爪...")
            self._send_gripper_command(1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        # 2. 移动到USB上方
        print("[MarvinUSBEnv] 移动到USB上方...")
        target_above = self.config.GRASP_POSE.copy()
        target_above[2] += 50  # Z方向上移50mm
        self._movla_to_xyzabc(target_above)
        time.sleep(0.5)

        # 3. 下降到USB位置
        print("[MarvinUSBEnv] 下降到USB位置...")
        self._movla_to_xyzabc(self.config.GRASP_POSE)
        time.sleep(0.5)

        # 4. 夹取USB
        if self.config.USE_GRIPPER:
            print("[MarvinUSBEnv] 夹取USB...")
            self._send_gripper_command(-1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

        # 5. 抬起USB
        print("[MarvinUSBEnv] 抬起USB...")
        self._update_currpos()
        lift_pose = self.config.GRASP_POSE.copy()
        lift_pose[2] += 30  # 抬起30mm
        self._movla_to_xyzabc(lift_pose)
        time.sleep(0.3)

        # 6. 移动到reset位置
        print("[MarvinUSBEnv] 移动到reset位置...")
        self._movla_to_xyzabc(self.config.RESET_POSE)
        time.sleep(0.5)

        # 7. 切换到柔顺模式
        print("[MarvinUSBEnv] 切换到柔顺模式...")
        self._switch_to_compliance_mode()

    def _manual_reset_usb(self):
        """手动复位: 机械臂归位 -> 等待人工放置USB"""
        print("[MarvinUSBEnv] 手动复位模式: 机械臂归位...")

        # 1. 机械臂回到 RESET_POSE
        self.go_to_reset(joint_reset=False)

        # 2. 打开夹爪 (如果USB已抓在手中则释放)
        if self.config.USE_GRIPPER:
            print("[MarvinUSBEnv] 打开夹爪...")
            self._send_gripper_command(1.0)
            time.sleep(self.config.GRIPPER_SLEEP)

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
