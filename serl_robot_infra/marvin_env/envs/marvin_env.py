"""
Marvin机械臂Gym环境实现

完全兼容HIL-SERL框架的Marvin机器人环境
设计参考: franka_env/envs/franka_env.py

重要约定:
- 欧拉角统一使用 scipy Rotation.from_euler("xyz", ..., degrees=False) / as_euler("xyz")
  Marvin 不使用 Franka 的 np.pi - yaw 偏移
- 单位: config 中用 mm/deg, 内部观测用 m/四元数
- 动作缩放参考 spacemouse_control 案例
- Ctrl+C 急停: 自动下使能 + 释放连接
"""
import os
import sys
import signal
import numpy as np
import gymnasium as gym
import cv2
import copy
import time
from scipy.spatial.transform import Rotation
from collections import OrderedDict
from typing import Dict

# 导入Marvin SDK
current_dir = os.path.dirname(os.path.abspath(__file__))
sdk_path = os.path.join(os.path.dirname(current_dir), 'SDK_PYTHON')
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from fx_robot import Marvin_Robot, DCSS
from fx_kine import Marvin_Kine

# 导入相机模块（复用FrankaEnv的相机实现）
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture

# 导入音频通知器
from franka_env.utils.audio_utils import get_audio_notifier


# ==============================================================================
# 标准欧拉角/四元数转换 (Marvin 不使用 Franka 的 np.pi - yaw 偏移)
# ==============================================================================

def _euler_to_quat(euler_rad):
    """欧拉角(弧度) -> 四元数 [x,y,z,w] (scipy 约定)"""
    return Rotation.from_euler("xyz", euler_rad).as_quat()


def _quat_to_euler(quat):
    """四元数 -> 欧拉角(弧度) (scipy 约定: intrinsic XYZ)"""
    return Rotation.from_quat(quat).as_euler("xyz")


##############################################################################


class MarvinEnv(gym.Env):
    """
    Marvin机械臂Gym环境

    设计原则：
    1. 观测空间和动作空间与FrankaEnv完全一致
    2. 使用Marvin SDK替代ROS通信
    3. 支持笛卡尔增量控制
    4. 支持阻抗控制模式
    """

    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config=None,
    ):
        """
        初始化Marvin环境

        Args:
            hz: 控制频率（Hz）
            fake_env: 是否为虚拟环境（测试用）
            save_video: 是否保存视频
            config: 环境配置对象
        """
        # ==================== 基础配置 ====================
        self.config = config
        self.hz = hz
        self.save_video = save_video
        self.fake_env = fake_env

        # 动作缩放参数 (参考 spacemouse: 平移 mm/step, 旋转 rad/step)
        self.action_scale = config.ACTION_SCALE

        # 任务参数 (config 中单位为 mm 和 deg)
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD

        # Episode管理
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.curr_path_length = 0

        # 显示和录制
        self.display_image = config.DISPLAY_IMAGE
        if self.save_video:
            self.recording_frames = []

        # 重置配置
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE  # mm
        self.random_rz_range = config.RANDOM_RZ_RANGE  # rad

        # ==================== 抖动检测 ====================
        self.last_joints = None  # 用于检测关节突变

        # ==================== 音频通知器 ====================
        self.audio_notifier = get_audio_notifier(
            device="plughw:3,0",
            enabled=True  # 设置为 False 可以禁用音频
        )

        # ==================== 定义Gym空间 ====================
        # 笛卡尔空间边界 (mm 和 deg)
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        # 动作空间: [Δx, Δy, Δz, Δrz, gripper]
        # 只保留 Z 轴旋转 (Δrz), X/Y 轴旋转由硬件锁定
        # 注意: 经过 RelativeFrame 变换后进入 step(), 单位已是基座系
        self.action_space = gym.spaces.Box(
            np.ones((5,), dtype=np.float32) * -1,
            np.ones((5,), dtype=np.float32),
        )

        # 观测空间: 与FrankaEnv保持一致
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "last_action": gym.spaces.Box(-1, 1, shape=(5,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(0, 255, shape=(256, 256, 3), dtype=np.uint8)
                        for key in config.REALSENSE_CAMERAS
                    }
                ),
            }
        )

        if fake_env:
            return

        # ==================== 初始化Marvin SDK ====================
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        self.kk = Marvin_Kine()
        self.arm = config.ARM
        self.arm_idx = 0 if self.arm == 'A' else 1

        # 连接机器人
        print(f"[MarvinEnv] 连接机器人: {config.ROBOT_IP}")
        ret = self.robot.connect(config.ROBOT_IP)
        if ret == 0:
            raise RuntimeError(f"Failed to connect to Marvin robot at {config.ROBOT_IP}")
        print("[MarvinEnv] 连接成功")

        time.sleep(0.5)

        # 初始化运动学库
        print(f"[MarvinEnv] 加载运动学配置: {config.KINE_CONFIG_PATH}")
        self.kk.log_switch(0)  # 关闭运动学日志
        ini_result = self.kk.load_config(
            arm_type=self.arm_idx,
            config_path=config.KINE_CONFIG_PATH
        )
        if not ini_result:
            raise RuntimeError(f"Failed to load kinematics config: {config.KINE_CONFIG_PATH}")

        initial_tag = self.kk.initial_kine(
            robot_type=ini_result['TYPE'][self.arm_idx],
            dh=ini_result['DH'][self.arm_idx],
            pnva=ini_result['PNVA'][self.arm_idx],
            j67=ini_result['BD'][self.arm_idx]
        )
        if not initial_tag:
            raise RuntimeError("Failed to initialize kinematics")
        print("[MarvinEnv] 运动学初始化成功")

        # 设置工具参数（重力补偿）
        # 只要配置了 TOOL_DYN_PARAMS 就调用 set_tool
        if np.any(config.TOOL_DYN_PARAMS != 0) or np.any(config.TOOL_KINE_PARAMS != 0):
            print("[MarvinEnv] 设置工具参数（重力补偿）...")
            self._set_tool_params()

        # 初始化相机
        self.cap = None
        self.init_cameras(config.REALSENSE_CAMERAS)
        print("[MarvinEnv] 相机初始化完成")

        self.curr_gripper_pos = 0.0
        self.last_action = np.zeros(5, dtype=np.float32)  # [dx,dy,dz,dry,gripper]

        # 夹爪时间控制
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.last_gripper_act = time.time()

        # 重置位姿 (mm/deg -> m/quat, 用于父类 go_to_reset)
        self.resetpos = np.concatenate([
            config.RESET_POSE[:3] / 1000.0,
            _euler_to_quat(np.deg2rad(config.RESET_POSE[3:]))
        ])

        # 随机重置配置 (mm -> m 转换在 go_to_reset 的 interpolate_move 输入时处理)
        self._random_xy_range_m = config.RANDOM_XY_RANGE / 1000.0
        self._random_rz_range_rad = config.RANDOM_RZ_RANGE

        # 周期性关节重置
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD

        # 初始化夹爪控制器
        self.gripper = None
        if hasattr(config, 'USE_GRIPPER') and config.USE_GRIPPER:
            try:
                from marvin_env.gripper.marvin_gripper import MarvinGripperController
                self.gripper = MarvinGripperController(
                    robot=self.robot,
                    arm=self.arm,
                    motor_id=config.GRIPPER_MOTOR_ID if hasattr(config, 'GRIPPER_MOTOR_ID') else 1
                )
                self.gripper.initialize()
                print("[MarvinEnv] 夹爪控制器已初始化")
            except Exception as e:
                print(f"[MarvinEnv] 夹爪初始化失败: {e}")
                self.gripper = None

        # 更新当前位置
        self._update_currpos()

        # ==================== 注册 Ctrl+C 急停 ====================
        self._register_emergency_stop()

        # ==================== 自动进入阻抗模式 ====================
        # 根据配置选择笛卡尔阻抗或关节阻抗
        self.impedance_mode = getattr(config, 'IMPEDANCE_MODE', 'cartesian')
        if self.impedance_mode == 'joint':
            print(f"[MarvinEnv] 使用关节阻抗模式 (IMPEDANCE_MODE={self.impedance_mode})")
            self._enter_joint_compliance_mode()
        else:
            print(f"[MarvinEnv] 使用笛卡尔阻抗模式 (IMPEDANCE_MODE={self.impedance_mode})")
            self._enter_compliance_mode()

        print("[MarvinEnv] 初始化完成")

    def init_cameras(self, name_serial_dict=None):
        """
        初始化相机（复用FrankaEnv的相机模块）
        """
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        if name_serial_dict is None:
            name_serial_dict = {}
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(RSCapture(name=cam_name, **kwargs))
            self.cap[cam_name] = cap

    # ==========================================================================
    # 插值移动 (参考 spacemouse_control 的 movLA 执行方式)
    # ==========================================================================

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """
        线性插值移动到目标位姿。

        Args:
            goal: 目标位姿:
                  - 6维 [x, y, z, rx, ry, rz] (m, rad)
                  - 7维 [x, y, z, qx, qy, qz, qw] (m, quat)
            timeout: 期望运动时间（秒）
        """
        self._update_currpos()
        current_xyzabc = self.currpos_xyzabc.copy()  # mm, deg

        # 统一转换为 XYZABC (mm, deg) 给 movLA
        if goal.shape == (6,):
            # [x,y,z m, rx,ry,rz rad] -> [mm, deg]
            euler_deg = np.rad2deg(goal[3:])
            target_xyzabc = np.concatenate([goal[:3] * 1000.0, euler_deg])
        elif goal.shape == (7,):
            # [x,y,z m, qx,qy,qz,qw] -> [mm, deg]
            euler_rad = _quat_to_euler(goal[3:])
            euler_deg = np.rad2deg(euler_rad)
            target_xyzabc = np.concatenate([goal[:3] * 1000.0, euler_deg])
        else:
            raise ValueError(f"Invalid goal shape: {goal.shape}")

        print(f"[interpolate_move] {current_xyzabc[:3]} -> {target_xyzabc[:3]} (mm)")

        try:
            points, _ = self.kk.movLA(
                start_xyzabc=current_xyzabc.tolist(),
                end_xyzabc=target_xyzabc.tolist(),
                ref_joints=self.curr_joints.tolist(),
                vel=100,
                acc=100,
                freq_hz=100
            )

            if not points or len(points) == 0:
                print("[interpolate_move] 轨迹规划失败，保持当前位置")
                return

            # 执行轨迹 (参考 spacemouse: 逐点 send_cmd, 无 sleep)
            for pt in points:
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                self.robot.send_cmd()
                # 无 sleep, 由底层缓冲处理

            # 更新状态
            self._update_currpos()

            # 更新 nextpos (7-DOF quat 格式)
            if goal.shape == (6,):
                self.nextpos = np.concatenate([
                    goal[:3], _euler_to_quat(goal[3:])
                ])
            else:
                self.nextpos = goal.copy()

            print(f"[interpolate_move] 完成, 当前位置: {self.currpos[:3]}")

        except Exception as e:
            print(f"[interpolate_move] 轨迹规划失败: {e}，回退到不移动")
            self._update_currpos()

    def close_cameras(self):
        """关闭所有相机"""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"[MarvinEnv] 关闭相机失败: {e}")

    # ==========================================================================
    # 状态读取 (参考 spacemouse_control._update_state)
    # ==========================================================================

    def _update_currpos(self):
        """
        更新当前机器人状态

        从Marvin机器人读取关节角度，通过正运动学计算末端位姿
        """
        # 订阅机器人数据
        sub_data = self.robot.subscribe(self.dcss)

        # 读取关节角度（单位：度）
        self.curr_joints = np.array(sub_data['outputs'][self.arm_idx]['fb_joint_pos'])

        # 🔍 DEBUG: 打印关节角度
        if self.curr_path_length == 0:  # 只在reset时打印
            print(f"\n{'='*70}")
            print(f"[FK_DEBUG] 正运动学计算开始")
            print(f"{'='*70}")
            print(f"[FK_DEBUG] curr_joints (度): {self.curr_joints}")

        # 正运动学：关节角度 -> 末端位姿 (mm, deg)
        fk_mat = self.kk.fk(joints=self.curr_joints.tolist())

        # 🔍 DEBUG: 打印FK返回的矩阵
        if self.curr_path_length == 0:
            print(f"[FK_DEBUG] fk_mat type: {type(fk_mat)}")
            fk_mat_array = np.array(fk_mat) if not isinstance(fk_mat, np.ndarray) else fk_mat
            print(f"[FK_DEBUG] fk_mat shape: {fk_mat_array.shape}")
            print(f"[FK_DEBUG] fk_mat:\n{fk_mat_array}")

        self.currpos_xyzabc = np.array(self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat))

        # 🔍 DEBUG: 打印转换后的xyzabc
        if self.curr_path_length == 0:
            print(f"[FK_DEBUG] currpos_xyzabc (mm,deg): {self.currpos_xyzabc}")
            print(f"[FK_DEBUG]   位置XYZ(mm): {self.currpos_xyzabc[:3]}")
            print(f"[FK_DEBUG]   姿态ABC(度): {self.currpos_xyzabc[3:]}")

        # 转换为四元数表示 [x, y, z, qx, qy, qz, qw] (m, quat)
        # 使用标准 scipy 转换，不使用 Franka 的 np.pi - yaw 偏移
        euler_rad = np.deg2rad(self.currpos_xyzabc[3:])
        quat = _euler_to_quat(euler_rad)

        # 🔍 DEBUG: 验证四元数转换
        quat_norm = np.linalg.norm(quat)
        if self.curr_path_length == 0:
            print(f"[FK_DEBUG]   欧拉角(弧度): {euler_rad}")
            print(f"[FK_DEBUG]   四元数: {quat}")
            print(f"[FK_DEBUG]   四元数模长: {quat_norm:.10f} {'✅' if abs(quat_norm - 1.0) < 0.01 else '❌ 异常！'}")

        if abs(quat_norm - 1.0) > 0.01:
            print(f"\n🚨 [QUAT_ERROR] 四元数未归一化！")
            print(f"  欧拉角(度): {self.currpos_xyzabc[3:]}")
            print(f"  欧拉角(弧度): {euler_rad}")
            print(f"  四元数: {quat}")
            print(f"  四元数模长: {quat_norm:.10f} (应该=1.0)")

            # 紧急修复：手动归一化（治标不治本）
            if quat_norm > 0.001:
                quat = quat / quat_norm
                print(f"  已强制归一化: {quat}")
            else:
                # 零向量无法归一化，使用单位四元数
                print(f"  四元数为零向量！使用单位四元数 [0,0,0,1]")
                quat = np.array([0, 0, 0, 1])

        self.currpos_quat = np.concatenate([
            self.currpos_xyzabc[:3] / 1000.0,
            quat
        ])

        if self.curr_path_length == 0:
            print(f"[FK_DEBUG] 最终currpos_quat: {self.currpos_quat}")
            print(f"[FK_DEBUG]   位置(m): {self.currpos_quat[:3]}")
            print(f"[FK_DEBUG]   四元数: {self.currpos_quat[3:]}")
            print(f"{'='*70}\n")

        # 兼容FrankaEnv的变量名
        self.currpos = self.currpos_quat

        # 读取夹爪位置
        if self.gripper is not None:
            try:
                old_gripper_pos = self.curr_gripper_pos
                raw_position = self.gripper.current_position  # 🔍 读取原始弧度值
                self.curr_gripper_pos = self.gripper.get_normalized_position()
                # 🔍 DEBUG: 仅在夹爪位置变化时打印（显示原始弧度值）
                if abs(self.curr_gripper_pos - old_gripper_pos) > 0.05:
                    print(f"[GRIPPER_DEBUG] 位置更新: 归一化 {old_gripper_pos:.3f} -> {self.curr_gripper_pos:.3f}, "
                          f"原始={raw_position:.3f}rad (OPEN={self.gripper.OPEN_RAD:.1f}, CLOSE={self.gripper.CLOSE_RAD:.1f})")
            except Exception as e:
                print(f"[GRIPPER_DEBUG] ⚠️ 读取夹爪位置失败: {e}")
                self.curr_gripper_pos = 0.0
        else:
            self.curr_gripper_pos = 0.0

    # ==========================================================================
    # 急停
    # ==========================================================================

    def _emergency_stop(self, signum=None, frame=None):
        """急停: 下使能 + 释放连接（参考 demo08_impedance_with_emergency_stop）"""
        print("\n" + "=" * 60)
        print("🛑 检测到 Ctrl+C，执行急停...")
        print("=" * 60)
        try:
            if hasattr(self, 'robot') and self.robot is not None:
                self.robot.clear_set()
                self.robot.set_state(arm=self.arm, state=0)  # A臂下使能
                self.robot.set_state(arm='B', state=0)        # B臂下使能
                self.robot.send_cmd()
                time.sleep(0.3)
                print("[急停] ✅ 机器人已下使能（关节锁死）")
        except Exception as e:
            print(f"[急停] ⚠️ 下使能失败: {e}")

        try:
            if hasattr(self, 'gripper') and self.gripper is not None:
                self.gripper.shutdown()
                print("[急停] ✅ 夹爪已失能")
        except Exception:
            pass

        try:
            if hasattr(self, 'robot') and self.robot is not None:
                self.robot.release_robot()
                print("[急停] ✅ 连接已释放")
        except Exception:
            pass

        print("=" * 60)
        print("🛑 急停完成")
        print("=" * 60)
        os._exit(0)

    def _register_emergency_stop(self):
        """注册 Ctrl+C 急停信号处理"""
        signal.signal(signal.SIGINT, self._emergency_stop)
        print("[MarvinEnv] ✅ 急停已就绪 (Ctrl+C)")

    # ==========================================================================
    # 阻抗模式
    # ==========================================================================

    def _enter_compliance_mode(self):
        """自动进入笛卡尔阻抗模式（含重力补偿）"""
        print("[MarvinEnv] 自动进入笛卡尔阻抗模式...")

        # 先设置工具参数（重力补偿），在 set_state=3 之前调用
        self._set_tool_params()

        # 设置笛卡尔阻抗参数
        self.robot.clear_set()
        self.robot.set_cart_kd_params(
            arm=self.arm,
            K=self.config.COMPLIANCE_PARAM['K'].tolist(),
            D=self.config.COMPLIANCE_PARAM['D'].tolist(),
            type=2
        )
        self.robot.send_cmd()
        time.sleep(0.5)

        # 切换到扭矩模式 + 笛卡尔阻抗
        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)  # 扭矩模式
        self.robot.set_impedance_type(arm=self.arm, type=2)  # 笛卡尔阻抗
        self.robot.set_vel_acc(
            arm=self.arm,
            velRatio=self.config.VEL_RATIO,
            AccRatio=self.config.ACC_RATIO
        )
        self.robot.send_cmd()
        time.sleep(2.0)

        # 设置末端笛卡尔控制参数
        self._update_currpos()
        xyzabc = self.currpos_xyzabc
        if xyzabc is not None:
            self.robot.clear_set()
            self.robot.set_EefCart_control_params(
                arm=self.arm,
                fcType=1,
                CartCtrlPara=[xyzabc[3], xyzabc[4], xyzabc[5], 0, 0, 0, 0]
            )
            self.robot.send_cmd()
            time.sleep(0.5)

        print("[MarvinEnv] ✅ 笛卡尔阻抗模式已激活")

    def _enter_joint_compliance_mode(self):
        """进入关节阻抗模式（柔顺控制）"""
        print("[MarvinEnv] 自动进入关节阻抗模式...")

        # 先设置工具参数（重力补偿），在 set_state=3 之前调用
        self._set_tool_params()

        # 设置关节阻抗参数
        joint_param = getattr(self.config, 'JOINT_COMPLIANCE_PARAM', None)
        if joint_param is None:
            print("[MarvinEnv] ⚠️ 未找到JOINT_COMPLIANCE_PARAM，使用默认值")
            joint_param = {
                "K": np.array([12.0, 12.0, 12.0, 10.0, 9.0, 9.0, 7.0]),
                "D": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]),
            }

        self.robot.clear_set()
        self.robot.set_joint_kd_params(
            arm=self.arm,
            K=joint_param['K'].tolist(),
            D=joint_param['D'].tolist()
        )
        self.robot.set_vel_acc(
            arm=self.arm,
            velRatio=self.config.VEL_RATIO,
            AccRatio=self.config.ACC_RATIO
        )
        self.robot.send_cmd()
        time.sleep(0.5)

        # 切换到扭矩模式 + 关节阻抗
        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)  # 扭矩模式
        self.robot.set_impedance_type(arm=self.arm, type=1)  # 关节阻抗
        self.robot.send_cmd()
        time.sleep(2.0)

        # 验证切换结果
        sub_data = self.robot.subscribe(self.dcss)
        cur_state = sub_data['states'][self.arm_idx]['cur_state']
        imp_type = sub_data['inputs'][self.arm_idx]['imp_type']
        joint_k = sub_data['inputs'][self.arm_idx]['joint_k'][:]
        joint_d = sub_data['inputs'][self.arm_idx]['joint_d'][:]

        print(f"[MarvinEnv] 关节阻抗状态: state={cur_state}, imp_type={imp_type}")
        print(f"[MarvinEnv] 设置的刚度K: {[round(k, 1) for k in joint_k]}")
        print(f"[MarvinEnv] 设置的阻尼D: {[round(d, 2) for d in joint_d]}")

        if cur_state == 3 and imp_type == 1:
            print("[MarvinEnv] ✅ 关节阻抗模式已激活")
        else:
            print(f"[MarvinEnv] ⚠️ 状态异常: state={cur_state}, imp_type={imp_type}")

    def _switch_to_precision_mode(self):
        """切换到高刚度精密模式（用于 reset 移动）"""
        self.robot.clear_set()
        self.robot.set_cart_kd_params(
            arm=self.arm,
            K=self.config.PRECISION_PARAM['K'].tolist(),
            D=self.config.PRECISION_PARAM['D'].tolist(),
            type=2
        )
        self.robot.send_cmd()
        time.sleep(0.5)

    def _switch_to_compliance_mode(self):
        """切换到柔顺模式（根据配置选择笛卡尔阻抗或关节阻抗）"""
        if self.impedance_mode == 'joint':
            self._switch_to_joint_compliance_mode()
        else:
            self._switch_to_cartesian_compliance_mode()

    def _switch_to_cartesian_compliance_mode(self):
        """切换到笛卡尔柔顺模式（带扭矩模式切换 + 重力补偿）"""
        self._set_tool_params()

        self.robot.clear_set()
        self.robot.set_cart_kd_params(
            arm=self.arm,
            K=self.config.COMPLIANCE_PARAM['K'].tolist(),
            D=self.config.COMPLIANCE_PARAM['D'].tolist(),
            type=2
        )
        self.robot.send_cmd()
        time.sleep(0.5)

        # 切换到阻抗模式 (state=3, imp_type=2) - 带验证
        max_retries = 5
        for attempt in range(max_retries):
            self.robot.clear_set()
            self.robot.set_state(arm=self.arm, state=3)
            self.robot.set_impedance_type(arm=self.arm, type=2)
            self.robot.set_vel_acc(
                arm=self.arm,
                velRatio=self.config.VEL_RATIO,
                AccRatio=self.config.ACC_RATIO
            )
            self.robot.send_cmd()
            time.sleep(0.5)

            # 验证切换成功
            for check in range(10):  # 最多等待2秒 (10 * 0.2s)
                sub = self.robot.subscribe(self.dcss)
                cur_state = sub['states'][self.arm_idx]['cur_state']
                imp_type = sub['inputs'][self.arm_idx]['imp_type']

                if cur_state == 3 and imp_type == 2:
                    print(f"[_switch_to_cartesian_compliance_mode] ✓ 切换成功: state={cur_state}, imp_type={imp_type} (尝试{attempt+1}/{max_retries})")
                    break

                time.sleep(0.2)

            if cur_state == 3 and imp_type == 2:
                break

            print(f"[_switch_to_cartesian_compliance_mode] ✗ 切换失败: state={cur_state}, imp_type={imp_type}, 重试 {attempt+1}/{max_retries}")
            time.sleep(0.5)

        if cur_state != 3 or imp_type != 2:
            print(f"[_switch_to_cartesian_compliance_mode] ⚠️ 警告: {max_retries}次尝试后仍未切换到笛卡尔阻抗 (当前state={cur_state}, imp_type={imp_type})")

        # 设置末端笛卡尔控制参数 (参考 spacemouse.enter_compliance_mode step 3)
        self._update_currpos()
        xyzabc = self.currpos_xyzabc
        if xyzabc is not None:
            self.robot.clear_set()
            self.robot.set_EefCart_control_params(
                arm=self.arm,
                fcType=1,
                CartCtrlPara=[xyzabc[3], xyzabc[4], xyzabc[5], 0, 0, 0, 0]
            )
            self.robot.send_cmd()
            time.sleep(0.5)

    def _switch_to_joint_compliance_mode(self):
        """切换到关节柔顺模式（带扭矩模式切换 + 重力补偿）"""
        self._set_tool_params()

        # 设置关节阻抗参数
        joint_param = getattr(self.config, 'JOINT_COMPLIANCE_PARAM', None)
        if joint_param is None:
            print("[MarvinEnv] ⚠️ 未找到JOINT_COMPLIANCE_PARAM，使用默认值")
            joint_param = {
                "K": np.array([12.0, 12.0, 12.0, 10.0, 9.0, 9.0, 7.0]),
                "D": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]),
            }

        self.robot.clear_set()
        self.robot.set_joint_kd_params(
            arm=self.arm,
            K=joint_param['K'].tolist(),
            D=joint_param['D'].tolist()
        )
        self.robot.send_cmd()
        time.sleep(0.5)

        # 切换到阻抗模式 (state=3, imp_type=1) - 带验证
        max_retries = 5
        for attempt in range(max_retries):
            self.robot.clear_set()
            self.robot.set_state(arm=self.arm, state=3)
            self.robot.set_impedance_type(arm=self.arm, type=1)
            self.robot.set_vel_acc(
                arm=self.arm,
                velRatio=self.config.VEL_RATIO,
                AccRatio=self.config.ACC_RATIO
            )
            self.robot.send_cmd()
            time.sleep(0.5)

            # 验证切换成功
            for check in range(10):  # 最多等待2秒 (10 * 0.2s)
                sub = self.robot.subscribe(self.dcss)
                cur_state = sub['states'][self.arm_idx]['cur_state']
                imp_type = sub['inputs'][self.arm_idx]['imp_type']

                if cur_state == 3 and imp_type == 1:
                    print(f"[_switch_to_joint_compliance_mode] ✓ 切换成功: state={cur_state}, imp_type={imp_type} (尝试{attempt+1}/{max_retries})")
                    break

                time.sleep(0.2)

            if cur_state == 3 and imp_type == 1:
                break

            print(f"[_switch_to_joint_compliance_mode] ✗ 切换失败: state={cur_state}, imp_type={imp_type}, 重试 {attempt+1}/{max_retries}")
            time.sleep(0.5)

        if cur_state != 3 or imp_type != 1:
            print(f"[_switch_to_joint_compliance_mode] ⚠️ 警告: {max_retries}次尝试后仍未切换到关节阻抗 (当前state={cur_state}, imp_type={imp_type})")


    def _set_tool_params(self):
        """设置工具参数（重力补偿），必须在 set_state=3（扭矩模式）之前调用。"""
        kine = self.config.TOOL_KINE_PARAMS.tolist()
        dyn = self.config.TOOL_DYN_PARAMS.tolist()
        self.robot.clear_set()
        self.robot.set_tool(arm=self.arm, kineParams=kine, dynamicParams=dyn)
        self.robot.send_cmd()
        time.sleep(0.3)
        print(f"[MarvinEnv] 工具参数已设置 (kine={kine}, dyn={dyn[:4]}...)")

    # ==========================================================================
    # 机器人命令
    # ==========================================================================

    def _send_pos_command(self, target_joints: np.ndarray, recover: bool = False):
        """
        发送关节位置指令。

        Args:
            target_joints: 目标关节角度（deg）
            recover: 是否在发送前清除错误。逐点轨迹时传 False 避免 0.1s 延迟
        """
        if recover:
            self._recover()
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=self.arm, joints=target_joints.tolist())
        self.robot.send_cmd()

    def _recover(self):
        """清除机器人错误状态"""
        try:
            self.robot.check_error_and_clear(self.dcss)
            time.sleep(0.005)
        except Exception as e:
            print(f"[MarvinEnv] 错误恢复失败: {e}")

    def _send_gripper_command(self, pos: float, mode="binary"):
        """
        发送夹爪指令

        Args:
            pos: 夹爪动作 [-1, 1]，-1为关闭，1为打开
            mode: 控制模式，"binary"或"continuous"
        """
        if self.gripper is None:
            return

        if mode == "binary":
            try:
                # 🔍 DEBUG: 打印每次调用的状态（包含原始弧度值）
                time_since_last = time.time() - self.last_gripper_act
                raw_pos = self.gripper.current_position if self.gripper else 0.0
                print(f"[GRIPPER_DEBUG] 输入动作={pos:.2f}, 归一化位置={self.curr_gripper_pos:.2f}, "
                      f"原始位置={raw_pos:.3f}rad, 距上次={time_since_last:.3f}s")

                if (pos <= -0.5) and \
                   (self.curr_gripper_pos > 0.5) and \
                   (time.time() - self.last_gripper_act > self.gripper_sleep):

                    print(f"[GRIPPER_DEBUG] ✅ 执行关闭: pos={pos:.2f} <= -0.5, "
                          f"curr_pos={self.curr_gripper_pos:.2f} > 0.5, "
                          f"time_ok={time_since_last:.3f} > {self.gripper_sleep:.2f}")
                    self.gripper.close(blocking=False)
                    self.last_gripper_act = time.time()
                    # ⚠️ 暂时注释掉立即更新，等下一个 _update_currpos 从电机读取真实值
                    # self.curr_gripper_pos = 0.0
                    print(f"[GRIPPER_DEBUG] 关闭命令已发送（等待电机到位）")

                elif (pos >= 0.5) and \
                     (self.curr_gripper_pos < 0.5) and \
                     (time.time() - self.last_gripper_act > self.gripper_sleep):

                    print(f"[GRIPPER_DEBUG] ✅ 执行打开: pos={pos:.2f} >= 0.5, "
                          f"curr_pos={self.curr_gripper_pos:.2f} < 0.5, "
                          f"time_ok={time_since_last:.3f} > {self.gripper_sleep:.2f}")
                    self.gripper.open(blocking=False)
                    self.last_gripper_act = time.time()
                    # ⚠️ 暂时注释掉立即更新，等下一个 _update_currpos 从电机读取真实值
                    # self.curr_gripper_pos = 1.0
                    print(f"[GRIPPER_DEBUG] 打开命令已发送（等待电机到位）")

                else:
                    # 🔍 DEBUG: 打印为什么没有执行
                    if pos <= -0.5:
                        if self.curr_gripper_pos <= 0.5:
                            print(f"[GRIPPER_DEBUG] ❌ 跳过关闭: 已经是闭合状态 (curr={self.curr_gripper_pos:.2f})")
                        elif time_since_last <= self.gripper_sleep:
                            print(f"[GRIPPER_DEBUG] ❌ 跳过关闭: 冷却中 ({time_since_last:.3f}s < {self.gripper_sleep:.2f}s)")
                    elif pos >= 0.5:
                        if self.curr_gripper_pos >= 0.5:
                            print(f"[GRIPPER_DEBUG] ❌ 跳过打开: 已经是打开状态 (curr={self.curr_gripper_pos:.2f})")
                        elif time_since_last <= self.gripper_sleep:
                            print(f"[GRIPPER_DEBUG] ❌ 跳过打开: 冷却中 ({time_since_last:.3f}s < {self.gripper_sleep:.2f}s)")
                    else:
                        print(f"[GRIPPER_DEBUG] ⚪ 跳过: 动作在死区 (pos={pos:.2f})")
                    return

            except Exception as e:
                print(f"[MarvinEnv] 夹爪控制失败: {e}")

        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    # ==========================================================================
    # 安全
    # ==========================================================================

    def clip_safety_box(self, xyzabc: np.ndarray) -> np.ndarray:
        """限制笛卡尔位姿在安全边界内 (mm, deg).
        注意: 仅裁剪位置;姿态由 movLA IK 自行验证可达性,
        避免 FK 欧拉角分支与安全边界不在同一分支导致硬裁剪跳变."""
        xyzabc = xyzabc.copy()
        xyzabc[:3] = np.clip(xyzabc[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high)
        return xyzabc

    def _resample_trajectory(self, points: list, num_points: int) -> list:
        """
        将movLA规划的轨迹均匀重采样为固定点数

        Args:
            points: movLA返回的关节角度轨迹 [[j1,j2,...,j7], ...]
            num_points: 目标点数（不包含起点）

        Returns:
            重采样后的轨迹（包含起点和终点）
        """
        if not points or len(points) == 0:
            return []

        points_array = np.array(points)  # shape: (N, 7)
        N = len(points_array)

        if N == 1:
            # 只有一个点，直接返回
            return points

        # 线性插值：从起点到终点均匀采样num_points个点（包含终点）
        # 索引范围: [0, N-1]
        indices = np.linspace(0, N - 1, num_points)

        # 对每个关节独立插值
        resampled = []
        for idx in indices:
            if idx == int(idx):
                # 正好落在原始点上
                resampled.append(points[int(idx)])
            else:
                # 需要插值
                i_low = int(np.floor(idx))
                i_high = int(np.ceil(idx))
                alpha = idx - i_low  # 插值权重

                joint_interp = (1 - alpha) * points_array[i_low] + alpha * points_array[i_high]
                resampled.append(joint_interp.tolist())

        return resampled

    # ==========================================================================
    # Gym step 接口 (参考 spacemouse_control 的控制逻辑)
    # ==========================================================================

    def step(self, action: np.ndarray) -> tuple:
        """
        执行一步动作（标准Gym接口）

        action 已经过 RelativeFrame 变换（从EE系到基座系），
        单位约定: [:3] Δ位姿(m), [3:6] Δ旋转(rad), [6] 夹爪

        ⚠️ 关节阻抗模式下的动作映射:
        - 笛卡尔阻抗: 直接使用movLA规划笛卡尔空间轨迹
        - 关节阻抗: 通过IK将笛卡尔增量转为关节角度增量，直接发送关节指令

        🔧 折衷方案 (10Hz 分两次执行):
        - 如果 self.hz == 10: 内部执行两次 50ms 子步骤
        - 如果 self.hz == 20: 直接执行一次 50ms

        Returns:
            (observation, reward, done, truncated, info)
        """
        # self._recover()  # 尝试恢复错误状态
        start_time = time.time()

        # ---- 计时器 ----
        t = {"0_start": time.time()}

        # 保存原始 action 用于调试
        raw_action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high).copy()
        print(f"[STEP_DEBUG step={self.curr_path_length}] 原始动作: {raw_action}, 裁剪后动作: {action}")
        # 保存当前动作到观测中（供下一步使用）
        self.last_action = action.copy()

        # 🔧 10Hz 折衷方案: 分两次执行
        if self.hz == 10:
            # 位置和旋转动作分两次执行，夹爪动作保持完整
            half_action = action.copy()
            half_action[:4] = action[:4] / 2.0  # 只分割位置和旋转 [dx,dy,dz,dry]
            # half_action[4] 保持原值（夹爪）

            # 第 1 次: 执行前半段动作
            self._execute_sub_step(half_action, raw_action, t, sub_step=1, gripper_action=action[4])

            # 第 2 次: 执行后半段动作
            self._execute_sub_step(half_action, raw_action, t, sub_step=2, gripper_action=action[4])
        else:
            # 20Hz 或其他频率: 直接执行完整动作
            self._execute_sub_step(action, raw_action, t, sub_step=0, gripper_action=action[4])

        # ==================== 5. 更新状态并返回 ====================
        self._update_currpos()
        t["5a_update_currpos"] = time.time()
        obs = self._get_obs()
        t["5b_get_obs"] = time.time()

        # 🔍 DEBUG: 详细打印观测空间数据（每10步打印一次）
        if self.curr_path_length % 10 == 0:
            print(f"\n{'='*70}")
            print(f"[STEP_OBS_DEBUG step={self.curr_path_length}] 原始观测数据分析")
            print(f"{'='*70}")
            print(f"obs keys: {obs.keys()}")
            print(f"obs['state'] keys: {obs['state'].keys()}")
            for key, value in obs['state'].items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"         value={value}")
            print(f"{'='*70}\n")

        reward = self.compute_reward(obs)
        t["5c_reward"] = time.time()
        done = self.curr_path_length >= self.max_episode_length or reward

        if self.curr_path_length % 10 == 0 or done:
            current_pose = obs["state"]["tcp_pose"]
            delta_xyz = np.abs(current_pose[:3] * 1000.0 - self._TARGET_POSE[:3])
            current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
            target_rot = Rotation.from_euler("xyz", np.deg2rad(self._TARGET_POSE[3:])).as_matrix()
            diff_rot = current_rot.T @ target_rot
            diff_euler = np.abs(np.rad2deg(Rotation.from_matrix(diff_rot).as_euler("xyz")))
            print(f"[DEBUG step={self.curr_path_length:3d}] "
                  f"tcp_xyz={np.array2string(current_pose[:3], precision=4, suppress_small=True)} m | "
                  f"Δxyz={np.array2string(delta_xyz, precision=1)} mm | "
                  f"Δrot={np.array2string(diff_euler, precision=1)} deg | "
                  f"reward={reward} | done={done}")


        # ---- 每步计时汇总 ----
        t["6_return"] = time.time()
        dt_total = (t["6_return"] - t["0_start"]) * 1000
        dt_update = (t["1_update_currpos"] - t["0_start"]) * 1000
        dt_recover = (t["2a_recover"] - t["1_update_currpos"]) * 1000
        dt_movla = (t["2b_movLA"] - t["2a_recover"]) * 1000
        dt_send = (t["2c_send_joints"] - t["2b_movLA"]) * 1000
        dt_mode = (t["2d_mode_check"] - t["2c_send_joints"]) * 1000
        dt_grip = (t["3_gripper"] - t["2d_mode_check"]) * 1000
        dt_sleep = (t["4_sleep"] - t["3_gripper"]) * 1000
        dt_update2 = (t["5a_update_currpos"] - t["4_sleep"]) * 1000
        dt_obs = (t["5b_get_obs"] - t["5a_update_currpos"]) * 1000
        dt_reward = (t["5c_reward"] - t["5b_get_obs"]) * 1000
        dt_final = (t["6_return"] - t["5c_reward"]) * 1000
        print(f"[step={self.curr_path_length:3d}][TIMING] total={dt_total:.1f}ms | "
              f"update1={dt_update:.1f} recover={dt_recover:.1f} movLA={dt_movla:.1f} send={dt_send:.1f} "
              f"mode={dt_mode:.1f} grip={dt_grip:.1f} sleep={dt_sleep:.1f} "
              f"update2={dt_update2:.1f} obs={dt_obs:.1f} reward={dt_reward:.1f} final={dt_final:.1f}")

        # ==================== 音频提示 ====================
        if done:
            if reward:  # 任务成功
                self.audio_notifier.play_success(step=self.curr_path_length)
            else:  # 任务失败（超时）
                self.audio_notifier.play_failure(reason="timeout")

        return obs, int(reward), done, False, {"succeed": reward}

    def _execute_sub_step(self, action: np.ndarray, raw_action: np.ndarray, t: dict, sub_step: int, gripper_action: float):
        """
        执行一个子步骤 (规划 + 发送 + sleep 50ms)

        根据阻抗模式选择不同的控制策略:
        - 笛卡尔阻抗: movLA规划笛卡尔轨迹
        - 关节阻抗: IK转换为关节增量，直接发送关节指令

        Args:
            action: 动作增量 (已缩放，位置/旋转部分可能已减半)
            raw_action: 原始动作 (用于调试)
            t: 计时字典
            sub_step: 子步骤编号 (0=单次执行, 1=第一次, 2=第二次)
            gripper_action: 夹爪动作 (完整值，不减半)
        """
        if sub_step > 0:
            print(f"[10Hz->20Hz] Sub-step {sub_step}/2: 执行{'前' if sub_step == 1 else '后'}半段动作...")

        # ==================== 1. 计算目标笛卡尔位姿 ====================
        self._update_currpos()
        t["1_update_currpos"] = time.time()
        current_xyzabc = self.currpos_xyzabc.copy()  # mm, deg

        # 位置增量: action[:3] 是基座系矢量, action_scale[0] 是 mm
        pos_delta_mm = action[:3] * self.action_scale[0]  # mm
        target_xyzabc = current_xyzabc.copy()
        target_xyzabc[:3] = current_xyzabc[:3] + pos_delta_mm

        # 姿态增量 (只处理 Z 轴旋转 drz, X/Y 由硬件锁定)
        fixed_orient = getattr(self.config, 'FIXED_ORIENTATION', False)
        if fixed_orient:
            action_rot_rad = np.zeros(3)
        else:
            action_rot_rad = np.zeros(3)
            # action[3] = drz (Z轴旋转)
            action_rot_rad[2] = action[3] * self.action_scale[1]  # 只保留 Z 轴旋转
        
        target_xyzabc[3:] = current_xyzabc[3:] + np.rad2deg(action_rot_rad)
        print(f"[step={self.curr_path_length}][sub_step={sub_step}] 当前位姿: {current_xyzabc}, "
              f"目标位姿增量: Δxyz={pos_delta_mm}, Δrot={np.rad2deg(action_rot_rad)}, 目标位姿: {target_xyzabc}")
        # 🔧 方案3: 强制锁定 A（X轴）和 B（Y轴）旋转到 RESET_POSE 的值
        # 只允许 C（Z轴）旋转变化
        target_xyzabc[3] = self._RESET_POSE[3]  # 强制锁定 A（X轴旋转）
        target_xyzabc[4] = self._RESET_POSE[4]  # 强制锁定 B（Y轴旋转）
        # target_xyzabc[5] (C, Z轴旋转) 允许变化

        # 安全限制
        target_xyzabc = self.clip_safety_box(target_xyzabc)

        # 保存目标位姿
        self.nextpos = np.concatenate([
            target_xyzabc[:3] / 1000.0,
            _euler_to_quat(np.deg2rad(target_xyzabc[3:]))
        ])

        # ==================== 2. 根据阻抗模式选择控制策略 ====================
        t["2a_recover"] = time.time()

        # 检查是否有足够大的位姿增量，避免 movLA 处理零长度轨迹时 C++ 层段错误
        delta_dist_mm = np.linalg.norm(pos_delta_mm)
        delta_rot_rad = np.linalg.norm(action_rot_rad)

        if self.impedance_mode == 'joint':
            # 关节阻抗模式：使用movLA规划 + 均匀重采样
            if delta_dist_mm > 0.01 or delta_rot_rad > 1e-7:
                try:
                    # 1. 用movLA规划轨迹（继承速度/加速度约束）
                    points, _ = self.kk.movLA(
                        start_xyzabc=current_xyzabc.tolist(),
                        end_xyzabc=target_xyzabc.tolist(),
                        ref_joints=self.curr_joints.tolist(),
                        vel=100,
                        acc=100,
                        freq_hz=100  # 使用100Hz规划
                    )

                    if points and len(points) > 0:
                        # 2. 均匀重采样为固定点数（20Hz环境 -> 10个点 = 200Hz）
                        NUM_INTERP_POINTS = 20
                        resampled_points = self._resample_trajectory(points, NUM_INTERP_POINTS)

                        print(f"[step={self.curr_path_length}][关节阻抗] movLA规划={len(points)}点, "
                              f"重采样={len(resampled_points)}点, 以200Hz发送...")

                        # 3. 逐点发送（200Hz = 5ms间隔）
                        t_send_start = time.time()
                        for pt in resampled_points:
                            self.robot.clear_set()
                            self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                            self.robot.send_cmd()
                            time.sleep(0.0025)  # 200Hz = 5ms

                        t_send_elapsed = (time.time() - t_send_start) * 1000
                        print(f"[step={self.curr_path_length}][关节阻抗] 发送完成: {len(resampled_points)}点, "
                              f"耗时={t_send_elapsed:.1f}ms")
                    else:
                        print(f"[step={self.curr_path_length}][关节阻抗] ⚠️ movLA规划失败，保持当前位置")
                except Exception as e:
                    print(f"[step={self.curr_path_length}][关节阻抗] ⚠️ 规划异常: {e}")
            else:
                print(f"[step={self.curr_path_length}][关节阻抗] 增量过小，跳过")
        else:
            # 笛卡尔阻抗模式：使用movLA规划轨迹
            if delta_dist_mm > 0.01 or delta_rot_rad > 1e-7:
                points, _ = self.kk.movLA(
                    start_xyzabc=current_xyzabc.tolist(),
                    end_xyzabc=target_xyzabc.tolist(),
                    ref_joints=self.curr_joints.tolist(),
                    vel=100,
                    acc=100,
                    freq_hz=100
                )
            else:
                points = None

            if points and len(points) > 0:
                # 🔍 DEBUG: 打印轨迹执行信息
                t_send_start = time.time()
                print(f"[step={self.curr_path_length}][笛卡尔阻抗] 规划={len(points)}点, 瞬间发送...")

                # 瞬间发送所有点（不加 sleep）
                for i, pt in enumerate(points):
                    self.robot.clear_set()
                    self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                    self.robot.send_cmd()

                t_send_elapsed = (time.time() - t_send_start) * 1000
                print(f"[step={self.curr_path_length}] 发送完成: {len(points)}点, 耗时={t_send_elapsed:.1f}ms")
            else:
                print(f"[MarvinEnv step={self.curr_path_length}] 警告: movLA失败 (delta_pos={delta_dist_mm:.3f}mm, delta_rot={delta_rot_rad:.5f}rad)")

        t["2b_movLA"] = time.time()
        t["2c_send_joints"] = time.time()

        # ---- 增强版状态打印：监控错误和关节误差 (每 5 步) ----
        if self.curr_path_length % 5 == 0:
            sub = self.robot.subscribe(self.dcss)
            cur_state = sub['states'][self.arm_idx]['cur_state']
            err_code = sub['states'][self.arm_idx]['err_code']  # ← 新增：错误码
            imp_type = sub['inputs'][self.arm_idx]['imp_type']

            # 读取关节反馈（实际位置 vs 指令位置）
            fb_joints = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])
            cmd_joints = np.array(sub['inputs'][self.arm_idx]['joint_cmd_pos'])  # ← 修复键名
            joint_error = cmd_joints - fb_joints  # 位置误差
            max_joint_error = np.max(np.abs(joint_error))

            mode_str = "关节阻抗" if self.impedance_mode == 'joint' else "笛卡尔阻抗"
            print(f"[step={self.curr_path_length:3d}][{mode_str}] state={cur_state}, err_code={err_code}, "
                  f"imp_type={imp_type}, max_joint_err={max_joint_error:.2f}°")

            # 🚨 如果检测到错误状态或异常误差
            if cur_state == 100 or err_code != 0:
                print(f"  ⚠️⚠️⚠️ [ALERT] 检测到错误状态！state={cur_state}, err_code={err_code}")
                print(f"  详细状态: {sub['states'][self.arm_idx]}")

            if max_joint_error > 5.0:
                print(f"  ⚠️ [WARNING] 关节误差过大: {max_joint_error:.2f}° > 5.0°")
                print(f"  误差分布: {np.array2string(joint_error, precision=2)}")
        t["2d_mode_check"] = time.time()

        # ==================== 3. 夹爪指令 ====================
        # 🔧 修复：每次 sub_step 都调用夹爪，确保状态一致
        # _send_gripper_command 内部已有防重复逻辑（状态检查 + 时间冷却）
        self._send_gripper_command(gripper_action)
        t["3_gripper"] = time.time()

        # ==================== 4. 自适应 sleep（确保20Hz节奏）====================
        # 计算从sub_step开始到现在的耗时
        elapsed_ms = (t["3_gripper"] - t["2a_recover"]) * 1000
        target_duration_ms = 50.0  # 20Hz = 50ms per step

        if elapsed_ms < target_duration_ms:
            sleep_time = (target_duration_ms - elapsed_ms) / 1000.0
            time.sleep(sleep_time)
            print(f"[step={self.curr_path_length}][自适应sleep] 已耗时={elapsed_ms:.1f}ms, 补充sleep={sleep_time*1000:.1f}ms")
        else:
            print(f"[step={self.curr_path_length}][⚠️超时] 已耗时={elapsed_ms:.1f}ms > 50ms，跳过sleep")

        t["4_sleep"] = time.time()

        # ==================== 5. 抖动检测：检测关节位置突变 ====================
        if self.last_joints is not None:
            joint_vel_sudden = np.abs(self.curr_joints - self.last_joints)
            max_sudden_change = np.max(joint_vel_sudden)

            if max_sudden_change > 5.0:  # 单步超过 5° 视为异常抖动
                print(f"\n🚨🚨🚨 [JITTER DETECTED at step={self.curr_path_length}]")
                sub = self.robot.subscribe(self.dcss)
                print(f"  cur_state: {sub['states'][self.arm_idx]['cur_state']}")
                print(f"  err_code: {sub['states'][self.arm_idx]['err_code']}")
                print(f"  imp_type: {sub['inputs'][self.arm_idx]['imp_type']}")
                print(f"  max_joint_change: {max_sudden_change:.2f}°")
                print(f"  joint_delta: {np.array2string(joint_vel_sudden, precision=2)}")
                print(f"  current_joints: {np.array2string(self.curr_joints, precision=1)}")
                print(f"  last_joints: {np.array2string(self.last_joints, precision=1)}")

                # 打印详细的状态信息
                print(f"  详细状态: {sub['states'][self.arm_idx]}")
                print(f"  详细输入: {sub['inputs'][self.arm_idx]}\n")

        # 更新上一步关节位置（用于下一次检测）
        self.last_joints = self.curr_joints.copy()

        # 只在完整步骤结束时增加计数器
        if sub_step == 0 or sub_step == 2:
            self.curr_path_length += 1

    # ==========================================================================
    # 奖励计算 (统一使用 scipy 标准转换)
    # ==========================================================================

    def compute_reward(self, obs) -> bool:
        """
        判断是否完成任务。

        当前姿态 (obs) 和目标姿态 (config.TARGET_POSE) 都使用
        标准 scipy Rotation 约定，确保一致性。
        """
        current_pose = obs["state"]["tcp_pose"]  # [x,y,z,qx,qy,qz,qw] (m, quat)

        # 位置误差 (mm)
        delta_xyz = np.abs(current_pose[:3] * 1000.0 - self._TARGET_POSE[:3])

        # 旋转误差: 两者都使用 from_quat / from_euler("xyz") 获取旋转矩阵
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", np.deg2rad(self._TARGET_POSE[3:])).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta_rot = np.abs(np.rad2deg(diff_euler))

        if np.all(delta_xyz < self._REWARD_THRESHOLD[:3]) and \
           np.all(delta_rot < self._REWARD_THRESHOLD[3:]):
            return True

        return False

    # ==========================================================================
    # 观测
    # ==========================================================================

    def _get_obs(self) -> dict:
        """获取当前观测"""
        t = {}
        t_start = time.time()

        t["get_im_start"] = time.time()
        images = self.get_im()
        t["get_im_end"] = time.time()

        t["subscribe_start"] = time.time()
        sub_data = self.robot.subscribe(self.dcss)
        t["subscribe_end"] = time.time()

        t["jacobian_start"] = time.time()
        # 计算末端速度（通过雅可比矩阵）
        joint_vel_deg = np.array(sub_data['outputs'][self.arm_idx]['fb_joint_vel'])  # deg/s
        jacobian = self.kk.joints2JacobMatrix(self.curr_joints.tolist())
        tcp_vel_sdk = jacobian @ joint_vel_deg  # [mm/s*3, deg/s*3]
        tcp_vel = np.concatenate([
            tcp_vel_sdk[:3] * 0.001,       # mm/s -> m/s
            np.deg2rad(tcp_vel_sdk[3:]),   # deg/s -> rad/s
        ])
        t["jacobian_end"] = time.time()

        t["force_start"] = time.time()
        # 读取末端力和力矩
        # est_cart_fn 始终为0（SDK未启用笛卡尔力估计）
        # 改用 est_joint_force + 雅可比矩阵转换: τ = JᵀF → F = (Jᵀ)⁺τ
        joint_force = np.array(sub_data['outputs'][self.arm_idx]['est_joint_force'])  # 7维
        jacobian = np.array(self.kk.joints2JacobMatrix(self.curr_joints.tolist()))  # 6x7
        cart_force_nm = np.linalg.pinv(jacobian.T) @ joint_force  # 通过雅可比静力转换
        tcp_force = cart_force_nm[:3]   # N
        tcp_torque = cart_force_nm[3:]  # Nm
        t["force_end"] = time.time()

        t["build_dict_start"] = time.time()
        state_observation = {
            "tcp_pose": self.currpos_quat,
            "tcp_vel": tcp_vel,
            "gripper_pose": np.array([self.curr_gripper_pos]),
            "tcp_force": tcp_force,
            "tcp_torque": tcp_torque,
            "last_action": self.last_action.copy(),
        }
        t["build_dict_end"] = time.time()

        t["deepcopy_start"] = time.time()
        result = copy.deepcopy(dict(images=images, state=state_observation))
        t["deepcopy_end"] = time.time()

        dt_im = (t["get_im_end"] - t["get_im_start"]) * 1000
        dt_sub = (t["subscribe_end"] - t["subscribe_start"]) * 1000
        dt_jac = (t["jacobian_end"] - t["jacobian_start"]) * 1000
        dt_force = (t["force_end"] - t["force_start"]) * 1000
        dt_build = (t["build_dict_end"] - t["build_dict_start"]) * 1000
        dt_dc = (t["deepcopy_end"] - t["deepcopy_start"]) * 1000
        dt_total = (time.time() - t_start) * 1000

        print(f"[_get_obs] get_im={dt_im:.1f}ms | subscribe={dt_sub:.1f}ms | "
              f"jacobian={dt_jac:.1f}ms | force={dt_force:.1f}ms | "
              f"build={dt_build:.1f}ms | deepcopy={dt_dc:.1f}ms | total={dt_total:.1f}ms")

        return result

    def get_im(self) -> Dict[str, np.ndarray]:
        """获取相机图像（避免重复读取共享相机）"""
        images = {}
        processed_caps = {}  # 记录已处理的相机对象

        for key, cap in self.cap.items():
            t0 = time.time()
            try:
                # 🔧 检查是否是共享相机（cap 对象相同）
                cap_id = id(cap)
                if cap_id in processed_caps:
                    # 复用已读取的图像
                    source_key = processed_caps[cap_id]
                    dt_reuse = (time.time() - t0) * 1000
                    print(f"[get_im] {key}: reuse from {source_key} ({dt_reuse:.1f}ms)")

                    # 直接复制处理后的图像（已经是 256x256 RGB）
                    images[key] = images[source_key].copy()
                    continue

                # 读取新的图像
                rgb = cap.read()
                dt_read = (time.time() - t0) * 1000
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(cropped_rgb, (256, 256))
                images[key] = resized[..., ::-1]  # BGR -> RGB
                dt_total = (time.time() - t0) * 1000
                print(f"[get_im] {key}: read={dt_read:.1f}ms total={dt_total:.1f}ms")

                # 记录此相机对象已处理
                processed_caps[cap_id] = key

                if self.save_video:
                    if not hasattr(self, 'recording_frames'):
                        self.recording_frames = []
                    self.recording_frames.append({key: cropped_rgb.copy()})

            except Exception as e:
                dt_total = (time.time() - t0) * 1000
                print(f"[get_im] {key} 读取失败 [{dt_total:.1f}ms]: {e}")
                images[key] = np.zeros((256, 256, 3), dtype=np.uint8)

        return images

    # ==========================================================================
    # Reset
    # ==========================================================================

    def reset(self, joint_reset=False, **kwargs):
        """重置环境（标准Gym接口）"""
        if self.curr_path_length != 0 and self.joint_reset_cycle != 0:
            if self.curr_path_length % self.joint_reset_cycle == 0:
                joint_reset = True
                print(f"[MarvinEnv] 周期性关节重置 (cycle={self.joint_reset_cycle})")

        print("\n[MarvinEnv] 重置环境...")

        if self.save_video:
            self.save_video_recording()

        # 错误恢复 -> 移动到重置位置 -> 错误恢复
        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()

        self.curr_path_length = 0
        self._update_currpos()

        print("[MarvinEnv] 重置完成\n")

        obs = self._get_obs()
        return obs, {"succeed": False}

    def go_to_reset(self, joint_reset=False):
        """
        移动到重置位置。

        (用 XYZABC (mm, deg) 作为中间表示, 避免四元数往返歧义)
        """
        print("[MarvinEnv] 执行go_to_reset...")

        # 1. 切换到精密模式
        self._update_currpos()
        current_joints = self.curr_joints.copy()
        self.robot.clear_set()
        self.robot.set_joint_cmd_pose(arm=self.arm, joints=current_joints.tolist())
        self.robot.send_cmd()
        time.sleep(0.3)

        self._switch_to_precision_mode()

        # 2. 关节重置 (暂不支持, 跳过)
        if joint_reset:
            print("[MarvinEnv] 关节重置暂不支持，跳过")

        # 3. 笛卡尔空间重置
        # 始终用 XYZABC (mm, deg) 表示, 避免四元数往返
        reset_xyzabc = self._RESET_POSE.copy()  # [X,Y,Z,A,B,C] mm, deg

        if self.randomreset:
            # XY 随机 (mm)
            reset_xyzabc[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            # 绕 Z 轴随机旋转 (度)
            reset_xyzabc[5] += np.rad2deg(np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            ))

            print(f"[MarvinEnv] 随机重置位置: {reset_xyzabc[:3]} (mm)")
        else:
            print(f"[MarvinEnv] 固定重置位置: {reset_xyzabc[:3]} (mm)")

        # 转换为 m, rad 传给 interpolate_move
        goal_6d = np.concatenate([
            reset_xyzabc[:3] / 1000.0,
            np.deg2rad(reset_xyzabc[3:])
        ])
        self.interpolate_move(goal_6d, timeout=1.0)

        # 4. 切换到柔顺模式
        self._switch_to_compliance_mode()

        self._update_currpos()
        print("[MarvinEnv] go_to_reset完成")

    def save_video_recording(self):
        """保存视频录制"""
        if not hasattr(self, 'recording_frames') or len(self.recording_frames) == 0:
            return

        try:
            print("[MarvinEnv] 保存视频...")
            self.recording_frames.clear()
        except Exception as e:
            print(f"[MarvinEnv] 保存视频失败: {e}")

    def close(self):
        """关闭环境，释放资源"""
        print("[MarvinEnv] 关闭环境...")

        if self.gripper is not None:
            try:
                self.gripper.shutdown()
            except Exception as e:
                print(f"[MarvinEnv] 夹爪关闭失败: {e}")

        self.close_cameras()

        # 正常退出前先下使能
        try:
            self.robot.clear_set()
            self.robot.set_state(arm=self.arm, state=0)
            self.robot.set_state(arm='B', state=0)
            self.robot.send_cmd()
            time.sleep(0.3)
        except Exception:
            pass
        self.robot.release_robot()

        print("[MarvinEnv] 已释放")
