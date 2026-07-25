#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Space Mouse 控制 Marvin 机械臂测试 - Test SpaceMouse Control

功能：
    通过 3Dconnexion Space Mouse 设备实时控制 Marvin 机械臂进行
    笛卡尔空间的平移和旋转操作，支持夹爪控制和急停保护。

主要功能：
    1. Space Mouse 设备初始化和读取
    2. 末端执行器坐标系到基座坐标系的变换
    3. 实时增量控制（位置和姿态）
    4. movLA 轨迹规划和平滑执行
    5. 夹爪开合控制
    6. 急停保护机制

工作流程：
    1. 初始化阶段：
       a) 连接设备：
          - 初始化 Space Mouse 设备
          - 连接 Marvin 机械臂
          - 初始化运动学引擎（Kine）
       b) 配置参数：
          - 设置阻抗控制参数
          - 配置动作缩放系数
          - 设置死区阈值
       c) 注册信号处理：
          - Ctrl+C 信号捕获
          - 急停处理函数

    2. 主控制循环：
       每个控制周期执行：

       a) 读取 Space Mouse 输入：
          - 调用 get_action() 获取增量
          - 返回格式: [dx, dy, dz, drx, dry, drz, button_states]
          - 坐标系：末端执行器坐标系
          - 映射关系：
            * SpaceMouse 物理轴 → 输出
            * 前/后（Y轴）→ -Y
            * 左/右（X轴）→ X
            * 上/下（Z轴）→ Z
            * Roll → -Roll
            * Pitch → -Pitch
            * Yaw → -Yaw

       b) 坐标变换：
          - 获取当前末端姿态（Rotation Matrix）
          - 将末端系增量变换到基座系：
            * 位置增量: delta_base = R_base_ee @ delta_ee
            * 姿态增量: 直接使用（欧拉角）
          - 应用死区阈值（过滤微小抖动）
          - 应用缩放系数（控制灵敏度）

       c) movLA 轨迹规划：
          - 计算目标关节位置：
            * 读取当前关节角度
            * 正运动学计算当前笛卡尔位姿
            * 加上基座系增量得到目标位姿
            * 逆运动学计算目标关节角度
          - 调用 MovLA_InPTPMode 规划轨迹：
            * 输入：起点关节、终点关节、速度、加速度
            * 输出：平滑的关节空间轨迹点序列
          - 记录规划点数和耗时

       d) 执行轨迹：
          - 遍历规划的轨迹点
          - 逐点调用 SetArmJoint 发送
          - 记录发送耗时

       e) 夹爪控制：
          - 检测按钮状态
          - Button 0：关闭夹爪
          - Button 1：打开夹爪

       f) 等待下一周期：
          - 计算已用时间
          - sleep 至控制周期结束

    3. 急停保护：
       - Ctrl+C 触发急停
       - 立即停止运动
       - 切换到下使能状态
       - 安全退出

数据流程图：
    ┌─────────────────────┐
    │ SpaceMouse 物理设备  │
    └──────────┬──────────┘
               │ 读取6DOF增量
               ↓
    ┌─────────────────────────────────────┐
    │ SpaceMouseExpert.get_action()        │
    │ 输出: [-y, x, z, -roll, -pitch, -yaw]│
    │ (末端执行器坐标系)                    │
    └──────────┬──────────────────────────┘
               │ 坐标变换
               ↓
    ┌─────────────────────────────────────┐
    │ _transform_spacemouse_to_base()      │
    │ - 旋转矩阵变换到基座系                │
    │ - 应用死区和缩放                      │
    └──────────┬──────────────────────────┘
               │ 基座系增量
               ↓
    ┌─────────────────────────────────────┐
    │ 计算目标位姿                          │
    │ - 当前位姿 + 增量 = 目标位姿           │
    │ - 逆运动学 → 目标关节角度             │
    └──────────┬──────────────────────────┘
               │ 目标关节
               ↓
    ┌─────────────────────────────────────┐
    │ MovLA_InPTPMode()                    │
    │ - 关节空间轨迹规划                    │
    │ - 输出平滑轨迹点序列                  │
    └──────────┬──────────────────────────┘
               │ 轨迹点序列
               ↓
    ┌─────────────────────────────────────┐
    │ 逐点发送 SetArmJoint()                │
    └──────────┬──────────────────────────┘
               │
               ↓
    ┌─────────────────────┐
    │   Marvin 机械臂      │
    └─────────────────────┘

配置说明：
    - ROBOT_IP: 机械臂 IP 地址
    - ARM: 机械臂编号（'A' 或 'B'）
    - KINE_CONFIG_PATH: 运动学配置文件路径
    - 控制参数：
      * CONTROL_HZ: 控制频率（默认 10Hz）
      * POS_SCALE: 位置缩放系数（mm/周期）
      * ROT_SCALE: 旋转缩放系数（rad/周期）
      * DEADBAND: 死区阈值（过滤抖动）
    - movLA 参数：
      * MOVLA_VEL: 速度参数
      * MOVLA_ACC: 加速度参数
      * MOVLA_FREQ_HZ: 规划频率

操作说明：
    Space Mouse 操作：
    - 推/拉帽子：沿末端执行器 X 轴平移
    - 左/右移动：沿末端执行器 Y 轴平移
    - 上/下移动：沿末端执行器 Z 轴平移
    - 倾斜帽子：绕对应轴旋转
    - 扭转帽子：绕 Z 轴旋转

    按钮控制：
    - 左按钮 (Button 0)：关闭夹爪
    - 右按钮 (Button 1)：打开夹爪

    退出：
    - Ctrl+C：急停并安全退出

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 运行控制程序
    python utils/test_tools/test_spacemouse_control.py

    # 操作 Space Mouse 控制机械臂
    # 按 Ctrl+C 停止

应用场景：
    1. 遥操作演示收集（teleoperation demonstration）
    2. 机械臂手动示教（manual teaching）
    3. 精细操作任务（pick-and-place）
    4. 人机交互界面测试
    5. 学习数据收集

注意事项：
    1. 确保 Space Mouse 设备已连接并识别
    2. 首次使用建议降低缩放系数（POS_SCALE, ROT_SCALE）
    3. 保持平滑操作，避免突然大幅度移动
    4. 注意工作空间边界，避免触碰限位
    5. 周围保持安全距离，准备好急停
    6. Ctrl+C 会立即锁死关节
    7. 末端执行器坐标系随机械臂姿态变化而变化
"""

import sys
import os
import time
import signal
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R

# 添加项目路径
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_file_dir)
# 项目根目录 (用于 serl_robot_infra 和 franka_env 导入)
sys.path.insert(0, os.path.join(_current_file_dir, '../..'))
# marvin_env 子目录 (用于 SDK_PYTHON 导入)
sys.path.insert(0, os.path.join(_current_file_dir, '../../serl_robot_infra/marvin_env'))
# serl_robot_infra 绝对路径 (用于 franka_env 子模块导入)
sys.path.insert(0, os.path.join(_current_file_dir, '../../serl_robot_infra'))


# ==============================================================================
# 配置参数
# ==============================================================================

# 机器人连接
ROBOT_IP = "192.168.14.190"
ARM = 'A'
KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

# Space Mouse 动作缩放
# pos_scale: 每个 spacemouse 单位对应的平移距离 (mm)
# rot_scale: 每个 spacemouse 单位对应的旋转角度 (rad)
# 与 config.py 的 ACTION_SCALE 对齐
POS_SCALE = 20.0      # 与 ACTION_SCALE[0] 一致
ROT_SCALE = 0.05      # 与 ACTION_SCALE[1] 一致 (原为0.015，改为0.05)

# 死区阈值 (忽略小于此值的输入)
DEADBAND = 0.015

# 控制频率 (Hz)
CONTROL_HZ = 20

# ==================== 阻抗模式选择 ====================
# "cartesian" - 笛卡尔阻抗（原始模式）
# "joint"     - 关节阻抗（新增：movLA规划 + 重采样）
IMPEDANCE_MODE = "cartesian"  # 切换为 "joint" 测试关节阻抗

# movLA 运动参数
MOVLA_VEL = 100   # 速度百分比 (关节阻抗推荐100)
MOVLA_ACC = 100   # 加速度百分比

# 关节阻抗重采样参数
NUM_INTERP_POINTS = 10  # 重采样点数（20Hz环境 -> 10点 = 200Hz发送）

# 安全边界 (XYZABC: mm 和 deg)
SAFETY_LOW = np.array([300.0, 50.0, 150.0, 160.0, -15.0, -105.0])
SAFETY_HIGH = np.array([700.0, 400.0, 550.0, 195.0, 15.0, -70.0])

# 夹爪电机 ID
GRIPPER_MOTOR_ID = 1


# ==============================================================================
# 测试类
# ==============================================================================

class SpaceMouseControlTest:
    """Space Mouse 控制 Marvin 机械臂测试"""

    def __init__(self):
        self.robot = None
        self.kk = None          # Marvin_Kine 运动学库
        self.dcss = None        # DCSS 数据结构
        self.gripper = None     # MarvinGripperController
        self.expert = None      # SpaceMouseExpert
        self.arm = ARM
        self.arm_idx = 0 if self.arm == 'A' else 1
        self.running = True

        # 当前状态缓存
        self.currpos_xyzabc = None   # [X,Y,Z,A,B,C] mm/deg
        self.currpos_quat = None     # [x,y,z,qx,qy,qz,qw] m/quat
        self.curr_joints = None      # 关节角度 (deg)

        # 调试开关
        self.debug = True  # 设为 True 打开详细变换日志

        # movLA 失败节流
        self._movla_fail_count = 0
        self._movla_fail_last_msg = ""

        # 记录初始位姿（用于显示相对偏移）
        self.initial_xyzabc = None

        # 阻抗模式
        self.impedance_mode = IMPEDANCE_MODE

        # 注册 Ctrl+C 信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

    # ==========================================================================
    # 安全与急停
    # ==========================================================================

    def signal_handler(self, sig, frame):
        """Ctrl+C 急停处理"""
        print("\n\n" + "=" * 60)
        print("🛑 检测到 Ctrl+C，执行急停...")
        print("=" * 60)
        self.emergency_stop()
        print("=" * 60)
        print("🛑 测试已终止")
        print("=" * 60)
        sys.exit(0)

    def emergency_stop(self):
        """急停: 关闭夹爪 → 下使能 → 释放资源"""
        # 1. 关闭 Space Mouse
        try:
            if self.expert is not None:
                self.expert.close()
                print("[急停] ✅ SpaceMouse 已关闭")
        except Exception:
            pass

        # 2. 关闭夹爪（保护末端执行器）
        try:
            if self.gripper is not None:
                self.gripper.shutdown()
                print("[急停] ✅ 夹爪已失能")
        except Exception:
            pass

        # 3. 下使能机器人
        try:
            if self.robot is not None:
                self.robot.clear_set()
                self.robot.set_state(arm=self.arm, state=0)   # 下使能当前臂
                self.robot.set_state(arm='B', state=0)         # 下使能另一臂
                self.robot.send_cmd()
                time.sleep(0.3)
                print("[急停] ✅ 机器人已下使能")
        except Exception as e:
            print(f"[急停] ⚠️ 下使能失败: {e}")

    # ==========================================================================
    # UI 辅助
    # ==========================================================================

    @staticmethod
    def print_section(title):
        """打印带格式的章节标题"""
        print("\n" + "=" * 60)
        print(f"📋 {title}")
        print("=" * 60)

    @staticmethod
    def wait_for_continue(prompt="按 Enter 继续，Ctrl+C 中止"):
        """等待用户确认"""
        print(f"\n{'─' * 60}")
        print(f"⏸️  {prompt}")
        print(f"{'─' * 60}")
        try:
            input()
        except KeyboardInterrupt:
            raise

    # ==========================================================================
    # 机器人连接与初始化
    # ==========================================================================

    def connect_robot(self):
        """连接 Marvin 机器人并初始化运动学"""
        self.print_section("连接 Marvin 机器人")

        from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
        from SDK_PYTHON.fx_kine import Marvin_Kine

        # 创建 SDK 对象
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        self.kk = Marvin_Kine()

        # 连接
        print(f"[连接] 目标 IP: {ROBOT_IP}")
        print(f"[连接] 使用臂: {self.arm} (idx={self.arm_idx})")
        ret = self.robot.connect(ROBOT_IP)
        if ret == 0:
            raise RuntimeError(f"无法连接到 {ROBOT_IP}")
        print("[连接] ✅ TCP 已连接")

        # 运动学初始化
        print(f"[运动学] 配置文件: {KINE_CONFIG_PATH}")
        self.kk.log_switch(0)
        ini = self.kk.load_config(arm_type=self.arm_idx, config_path=KINE_CONFIG_PATH)
        if not ini:
            raise RuntimeError("运动学配置加载失败")

        tag = self.kk.initial_kine(
            robot_type=ini['TYPE'][self.arm_idx],
            dh=ini['DH'][self.arm_idx],
            pnva=ini['PNVA'][self.arm_idx],
            j67=ini['BD'][self.arm_idx],
        )
        if not tag:
            raise RuntimeError("运动学初始化失败")
        print("[运动学] ✅ 初始化成功")

        # 获取初始状态
        time.sleep(0.3)
        self._update_state()
        self.initial_xyzabc = self.currpos_xyzabc.copy()
        print(f"[状态] 当前关节角: {np.array2string(self.curr_joints, precision=1)}")
        print(f"[状态] 当前 TCP (XYZABC): {np.array2string(self.currpos_xyzabc, precision=1)}")

        # 初始化夹爪
        self._init_gripper()

        print("[连接] ✅ 全部初始化完成")

    def _init_gripper(self):
        """初始化夹爪控制器"""
        self.print_section("初始化夹爪")
        try:
            from serl_robot_infra.marvin_env.gripper.marvin_gripper import MarvinGripperController
            self.gripper = MarvinGripperController(
                self.robot, self.arm, motor_id=GRIPPER_MOTOR_ID
            )
            if self.gripper.initialize():
                print("[夹爪] ✅ 已使能")
                # 初始打开夹爪
                self.gripper.open(blocking=True)
                print("[夹爪] ✅ 已打开")
            else:
                print("[夹爪] ⚠️ 初始化失败，测试中禁用夹爪")
                self.gripper = None
        except ImportError as e:
            print(f"[夹爪] ⚠️ 无法导入 MarvGripperController: {e}")
            print("[夹爪] ⚠️ 测试中禁用夹爪")
            self.gripper = None
        except Exception as e:
            print(f"[夹爪] ⚠️ 初始化异常: {e}")
            self.gripper = None

    # ==========================================================================
    # 柔顺模式
    # ==========================================================================

    def enter_compliance_mode(self):
        """进入阻抗模式（笛卡尔或关节）"""
        from serl_robot_infra.marvin_env.envs.config import DefaultMarvinEnvConfig
        config = DefaultMarvinEnvConfig()

        if self.impedance_mode == 'joint':
            self._enter_joint_compliance_mode(config)
        else:
            self._enter_cartesian_compliance_mode(config)

    def _enter_cartesian_compliance_mode(self, config):
        """进入笛卡尔阻抗模式"""
        self.print_section("进入笛卡尔阻抗模式")

        K = config.COMPLIANCE_PARAM['K'].tolist()
        D = config.COMPLIANCE_PARAM['D'].tolist()

        print(f"[阻抗] K: {K}")
        print(f"[阻抗] D: {D}")

        # Step 1: 设置笛卡尔阻抗参数
        self.robot.clear_set()
        self.robot.set_cart_kd_params(arm=self.arm, K=K, D=D, type=2)
        self.robot.send_cmd()
        time.sleep(0.5)
        print("[阻抗] ✅ 阻抗参数已设置")

        # Step 2: 切换到扭矩模式 + 笛卡尔阻抗
        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)             # 扭矩模式
        self.robot.set_impedance_type(arm=self.arm, type=2)      # 笛卡尔阻抗
        self.robot.set_vel_acc(arm=self.arm, velRatio=10, AccRatio=10)
        self.robot.send_cmd()
        time.sleep(2.0)
        print("[阻抗] ✅ 已切换到扭矩模式 + 笛卡尔阻抗")

        # Step 3: 设置末端笛卡尔控制参数
        self._update_state()
        xyzabc = self.currpos_xyzabc
        if xyzabc is not None:
            self.robot.clear_set()
            self.robot.set_EefCart_control_params(
                arm=self.arm,
                fcType=1,
                CartCtrlPara=[xyzabc[3], xyzabc[4], xyzabc[5], 0, 0, 0, 0],
            )
            self.robot.send_cmd()
            time.sleep(0.5)
            print("[阻抗] ✅ 末端控制参数已设置")

        print("[阻抗] ✅ 笛卡尔阻抗模式已激活")

    def _enter_joint_compliance_mode(self, config):
        """进入关节阻抗模式（与MarvinEnv._enter_joint_compliance_mode对齐）"""
        self.print_section("进入关节阻抗模式")

        # Step 1: 设置工具参数（重力补偿）
        tool_dyn = config.TOOL_DYN_PARAMS.tolist()
        tool_kine = config.TOOL_KINE_PARAMS.tolist()
        print(f"[阻抗] 工具动力学参数: {tool_dyn}")
        print(f"[阻抗] 工具运动学参数: {tool_kine}")

        self.robot.clear_set()
        self.robot.set_tool(arm=self.arm, kineParams=tool_kine, dynamicParams=tool_dyn)
        self.robot.send_cmd()
        time.sleep(0.3)
        print("[阻抗] ✅ 工具参数已设置")

        # Step 2: 设置关节刚度/阻尼参数
        K = config.JOINT_COMPLIANCE_PARAM['K'].tolist()
        D = config.JOINT_COMPLIANCE_PARAM['D'].tolist()
        print(f"[阻抗] 关节刚度 K: {K}")
        print(f"[阻抗] 关节阻尼 D: {D}")

        self.robot.clear_set()
        self.robot.set_joint_kd_params(arm=self.arm, K=K, D=D)
        self.robot.send_cmd()
        time.sleep(0.5)
        print("[阻抗] ✅ 关节阻抗参数已设置")

        # Step 3: 切换到扭矩模式 + 关节阻抗
        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)             # 扭矩模式
        self.robot.set_impedance_type(arm=self.arm, type=1)     # 关节阻抗
        self.robot.set_vel_acc(arm=self.arm, velRatio=50, AccRatio=50)
        self.robot.send_cmd()
        time.sleep(2.0)
        print("[阻抗] ✅ 已切换到扭矩模式 + 关节阻抗")

        # Step 4: 验证状态（处理切换中状态101）
        sub = self.robot.subscribe(self.dcss)
        cur_state = sub['states'][self.arm_idx]['cur_state']
        imp_type = sub['inputs'][self.arm_idx]['imp_type']

        # 如果状态是101（切换中），等待切换完成
        if cur_state == 101:
            print("[阻抗] ○ 模式切换中（state=101），等待...")
            time.sleep(2.0)
            sub = self.robot.subscribe(self.dcss)
            cur_state = sub['states'][self.arm_idx]['cur_state']
            imp_type = sub['inputs'][self.arm_idx]['imp_type']
            print(f"[阻抗] 切换完成: state={cur_state}, impedance_type={imp_type}")

        if cur_state == 3 and imp_type == 1:
            print("[阻抗] ✅ 关节阻抗模式已激活")
        else:
            print(f"[阻抗] ⚠️ 状态验证失败: state={cur_state}, impedance_type={imp_type}")
            print(f"[阻抗] 详细状态: {sub['states'][self.arm_idx]}")
            print(f"[阻抗] 详细输入: {sub['inputs'][self.arm_idx]}")

    # ==========================================================================
    # 状态读取
    # ==========================================================================

    def _update_state(self):
        """更新当前机器人状态（关节角 → FK → 笛卡尔位姿）"""
        sub = self.robot.subscribe(self.dcss)
        self.curr_joints = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])

        # 正运动学: 关节角 → 4x4 齐次矩阵 → XYZABC
        fk_mat = self.kk.fk(joints=self.curr_joints.tolist())
        self.currpos_xyzabc = np.array(self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat))

        # XYZABC (mm/deg) → quat pose (m/quat)
        from franka_env.utils.rotations import euler_2_quat
        xyz_m = self.currpos_xyzabc[:3] / 1000.0
        euler_rad = np.deg2rad(self.currpos_xyzabc[3:])
        quat = euler_2_quat(euler_rad)
        self.currpos_quat = np.concatenate([xyz_m, quat])

    # ==========================================================================
    # 坐标变换: 末端执行器系 → 基座系
    # ==========================================================================

    def _transform_spacemouse_to_base(self, action_ee: np.ndarray):
        """
        将 Space Mouse 的末端执行器系动作转换为基座系。

        Space Mouse 输出:
            action_ee = [-y, x, z, -roll, -pitch, -yaw]  # 末端执行器坐标系
            注意 y 和 x 是交换的，且符号有翻转（SpaceMouseExpert 内部处理）

        变换过程 (与 RelativeFrame.transform_action 对齐):
            pos_base  = R @ action_ee[:3]
            rot_base  = R @ action_ee[3:6]
            其中 R = Rotation.from_quat(currpos_quat[3:]).as_matrix()

        Args:
            action_ee: Space Mouse 原始输出 [y, x, z, roll, pitch, yaw]

        Returns:
            (pos_delta_mm, rot_delta_rad): 基座系下的增量 (mm, rad)
        """
        # 当前末端执行器旋转矩阵 (3x3)
        rot_matrix = R.from_quat(self.currpos_quat[3:]).as_matrix()

        # 平移: 末端执行器系 → 基座系，缩放为 mm
        pos_ee = action_ee[:3]                       # [y, x, z] in EE frame
        pos_base = rot_matrix @ pos_ee                # rotate to base frame
        pos_delta_mm = pos_base * POS_SCALE           # scale to mm

        # 旋转: 末端执行器系 → 基座系，缩放为 rad
        rot_ee = action_ee[3:6]                       # [roll, pitch, yaw] in EE frame
        rot_base = rot_matrix @ rot_ee                # rotate to base frame
        rot_delta_rad = rot_base * ROT_SCALE          # scale to radians

        if self.debug:
            print(f"\n[变换] ──────────────────────────────────")
            print(f"  SM原始:            {np.array2string(action_ee, precision=4, suppress_small=True)}")
            print(f"  当前TCP (XYZABC):  {np.array2string(self.currpos_xyzabc, precision=2, suppress_small=True)}")
            print(f"  当前TCP (quat):    {np.array2string(self.currpos_quat, precision=4, suppress_small=True)}")
            print(f"  旋转矩阵 R:        {np.array2string(rot_matrix, precision=3, suppress_small=True).replace(chr(10), chr(10) + ' ' * 22)}")
            print(f"  pos_ee:            {np.array2string(pos_ee, precision=4, suppress_small=True)}")
            print(f"  pos_base (R@ee):   {np.array2string(pos_base, precision=4, suppress_small=True)}")
            print(f"  pos_delta (mm):    {np.array2string(pos_delta_mm, precision=4, suppress_small=True)}")
            print(f"  rot_ee:            {np.array2string(rot_ee, precision=4, suppress_small=True)}")
            print(f"  rot_base (R@ee):   {np.array2string(rot_base, precision=4, suppress_small=True)}")
            print(f"  rot_delta (rad):   {np.array2string(rot_delta_rad, precision=4, suppress_small=True)}")

        return pos_delta_mm, rot_delta_rad

    # ==========================================================================
    # 轨迹重采样（与MarvinEnv对齐）
    # ==========================================================================

    def _resample_trajectory(self, points: list, num_points: int) -> list:
        """
        将movLA规划的轨迹均匀重采样为固定点数

        Args:
            points: movLA返回的关节角度轨迹 [[j1,j2,...,j7], ...]
            num_points: 目标点数

        Returns:
            重采样后的轨迹（包含起点和终点）
        """
        if not points or len(points) == 0:
            return []

        points_array = np.array(points)  # shape: (N, 7)
        N = len(points_array)

        if N == 1:
            return points

        # 线性插值：从起点到终点均匀采样num_points个点
        indices = np.linspace(0, N - 1, num_points)

        resampled = []
        for idx in indices:
            if idx == int(idx):
                # 正好落在原始点上
                resampled.append(points[int(idx)])
            else:
                # 需要插值
                i_low = int(np.floor(idx))
                i_high = int(np.ceil(idx))
                alpha = idx - i_low

                joint_interp = (1 - alpha) * points_array[i_low] + alpha * points_array[i_high]
                resampled.append(joint_interp.tolist())

        return resampled

    # ==========================================================================
    # 安全移动
    # ==========================================================================

    def _safe_move(self, pos_delta_mm: np.ndarray, rot_delta_rad: np.ndarray):
        """
        基于增量安全移动到目标位姿。

        姿态使用欧拉角直接加法（每步增量 < 0.01 rad，加法逼近旋转合成足够准确），
        避免四元数往返 → 欧拉角分支跳变（如 A: -98° → 160° 导致 IK 失败）。

        Args:
            pos_delta_mm:  基座系位置增量 (mm)
            rot_delta_rad: 基座系旋转增量 (rad, 近似视为欧拉角增量)
        """
        current = self.currpos_xyzabc.copy()
        current_joints = self.curr_joints.tolist()

        for attempt in range(2):  # 最多重试一次（减半）
            # ---- 1. 位置增量 ----
            target = current.copy()
            target[:3] = current[:3] + pos_delta_mm

            # ---- 2. 姿态增量: 欧拉角直接加法 ----
            target[3:] = current[3:] + np.rad2deg(rot_delta_rad)

            # ---- 3. 安全边界裁剪 ----
            target[:3] = np.clip(target[:3], SAFETY_LOW[:3], SAFETY_HIGH[:3])

            # ---- 4. movLA 规划 ----
            points = None
            try:
                points, _ = self.kk.movLA(
                    start_xyzabc=current.tolist(),
                    end_xyzabc=target.tolist(),
                    ref_joints=current_joints,
                    vel=MOVLA_VEL,
                    acc=MOVLA_ACC,
                    freq_hz=100,  # 使用100Hz规划（与MarvinEnv对齐）
                )
            except Exception:
                if attempt == 0:
                    pos_delta_mm *= 0.5
                    rot_delta_rad *= 0.5
                    continue

            if not points or len(points) == 0:
                msg = (f"movLA 返回空 "
                       f"| 起点: {np.array2string(current, precision=1, suppress_small=True)}"
                       f"| 目标: {np.array2string(target, precision=1, suppress_small=True)}"
                       f"| Δ={np.array2string(pos_delta_mm, precision=2, suppress_small=True)}mm "
                       f"| joint6={current_joints[5]:.1f}°")
                if msg != self._movla_fail_last_msg:
                    print(f"[movLA FAIL#{self._movla_fail_count:03d}] {msg}")
                    self._movla_fail_last_msg = msg
                self._movla_fail_count += 1
                if attempt == 0:
                    pos_delta_mm *= 0.5
                    rot_delta_rad *= 0.5
                    continue
                return

            # ---- 5. 根据阻抗模式执行 ----
            if self.impedance_mode == 'joint':
                # 关节阻抗模式：重采样 + 200Hz发送（与MarvinEnv._execute_sub_step对齐）
                resampled_points = self._resample_trajectory(points, NUM_INTERP_POINTS)
                print(f"[movLA] 规划={len(points)}点, 重采样={len(resampled_points)}点 | "
                      f"Δ={np.array2string(pos_delta_mm, precision=2, suppress_small=True)}mm")

                for pt in resampled_points:
                    self.robot.clear_set()
                    self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                    self.robot.send_cmd()
                    time.sleep(0.005)  # 200Hz = 5ms
            else:
                # 笛卡尔阻抗模式：瞬间发送所有点
                print(f"[movLA] 规划点数: {len(points)} | Δ={np.array2string(pos_delta_mm, precision=2, suppress_small=True)}mm")
                for pt in points:
                    self.robot.clear_set()
                    self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                    self.robot.send_cmd()

            return  # 成功，退出

    # ==========================================================================
    # 夹爪控制
    # ==========================================================================

    def _handle_gripper(self, left_pressed: bool, right_pressed: bool,
                        prev_left: bool, prev_right: bool):
        """
        处理夹爪按钮（仅上升沿触发）。

        Args:
            left_pressed:  左按钮当前状态
            right_pressed: 右按钮当前状态
            prev_left:     左按钮上一状态
            prev_right:    右按钮上一状态
        """
        if self.gripper is None:
            return

        # 左按钮: 关闭夹爪 (上升沿)
        if left_pressed and not prev_left:
            print("  🔴 关闭夹爪")
            try:
                self.gripper.close(blocking=True)
            except Exception as e:
                print(f"  ⚠️ 关闭夹爪失败: {e}")

        # 右按钮: 打开夹爪 (上升沿)
        if right_pressed and not prev_right:
            print("  🟢 打开夹爪")
            try:
                self.gripper.open(blocking=True)
            except Exception as e:
                print(f"  ⚠️ 打开夹爪失败: {e}")

    # ==========================================================================
    # 主控循环
    # ==========================================================================

    def run(self):
        """运行 Space Mouse 控制测试"""
        # ---- 欢迎信息 ----
        print("\n" + "=" * 60)
        print("🚀 Space Mouse 控制 Marvin 机械臂测试")
        print("=" * 60)
        print()
        print("操作说明:")
        print("  推/拉 Space Mouse 帽:  沿末端执行器轴平移")
        print("  倾斜/扭转帽:          绕末端执行器轴旋转")
        print("  左按钮 (button 0):    关闭夹爪")
        print("  右按钮 (button 1):    打开夹爪")
        print("  Ctrl+C:               急停退出")
        print()
        print(f"阻抗模式: {self.impedance_mode.upper()}")
        if self.impedance_mode == 'joint':
            print(f"  → 关节阻抗: movLA规划 + 重采样{NUM_INTERP_POINTS}点 + 200Hz发送")
        else:
            print(f"  → 笛卡尔阻抗: movLA规划 + 瞬间发送所有点")
        print()
        print(f"缩放参数:")
        print(f"  平移: {POS_SCALE} mm/unit")
        print(f"  旋转: {ROT_SCALE:.3f} rad/unit (~{np.rad2deg(ROT_SCALE):.1f}°/unit)")
        print(f"  死区: {DEADBAND}")
        print(f"  控制频率: {CONTROL_HZ} Hz")
        print(f"  movLA速度/加速度: {MOVLA_VEL}/{MOVLA_ACC}")
        print(f"  安全边界 XYZ: {SAFETY_LOW[:3]} → {SAFETY_HIGH[:3]} (mm)")
        print("=" * 60)
        self.wait_for_continue("准备好后按 Enter 开始")

        try:
            # ---- Step 1: 连接机器人 ----
            self.connect_robot()

            # ---- Step 2: 进入柔顺模式 ----
            self.enter_compliance_mode()

            # ---- Step 3: 初始化 Space Mouse ----
            self.print_section("初始化 Space Mouse")
            from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
            self.expert = SpaceMouseExpert()
            time.sleep(0.3)  # 等待守护进程启动
            # 清空缓冲区
            action, _ = self.expert.get_action()
            print(f"[SpaceMouse] 初始读数: {np.array2string(action, precision=4)}")
            print("[SpaceMouse] ✅ 已就绪")

            # ---- Step 4: 控制循环 ----
            print("\n" + "=" * 60)
            print("✅ 准备就绪，开始 Space Mouse 控制！")
            print("   移动 Space Mouse 控制机械臂")
            print("   如果无法控制，在另一个终端运行: pkill -9 -f test_spacemouse")
            print("   然后检查机器人错误灯，按急停复位")
            print("   Ctrl+C 退出")
            print("=" * 60)

            prev_buttons = [0, 0]
            last_pose_print = time.time()
            error_check_counter = 0

            while self.running:
                loop_start = time.time()

                # ---- 4a. 读取 Space Mouse ----
                action, buttons = self.expert.get_action()

                # buttons 是 [left, right] 两按钮
                left_pressed = bool(buttons[0]) if len(buttons) > 0 else False
                right_pressed = bool(buttons[1]) if len(buttons) > 1 else False

                # ---- 4b. 夹爪控制 (上升沿触发) ----
                self._handle_gripper(
                    left_pressed, right_pressed,
                    prev_buttons[0], prev_buttons[1],
                )
                prev_buttons = [left_pressed, right_pressed]

                # ---- 4c. 运动控制 ----
                norm = np.linalg.norm(action)
                if norm > DEADBAND:
                    # 统一读取一次状态（减少 subscribe 次数）
                    self._update_state()

                    # 定期检查错误状态（每100次循环）
                    error_check_counter += 1
                    if error_check_counter >= 100:
                        error_check_counter = 0
                        sub = self.robot.subscribe(self.dcss)
                        cur_state = sub['states'][self.arm_idx]['cur_state']
                        error_code = sub['states'][self.arm_idx].get('error_code', 0)

                        if error_code != 0:
                            print(f"\n[⚠️ 错误] 检测到错误代码: {error_code}, state={cur_state}")
                            print("   请检查机器人状态，按急停复位后重新运行")
                        elif cur_state != 3:
                            print(f"\n[⚠️ 警告] 不在扭矩模式: state={cur_state}")

                    # 变换: 末端执行器系 → 基座系
                    pos_delta_mm, rot_delta_rad = \
                        self._transform_spacemouse_to_base(action)

                    # 执行移动
                    self._safe_move(pos_delta_mm, rot_delta_rad)

                    # 周期性打印状态 (每 2 秒)
                    now = time.time()
                    if now - last_pose_print > 2.0:
                        delta_since_start = self.currpos_xyzabc - self.initial_xyzabc
                        print(f"  TCP: {np.array2string(self.currpos_xyzabc, precision=1)}"
                              f" | Δ: {np.array2string(delta_since_start, precision=1)}"
                              f" | SM: {np.array2string(action, precision=3)}")
                        last_pose_print = now

                # ---- 4d. 控制频率 ----
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0 / CONTROL_HZ - elapsed)
                if sleep_time > 0:
                    # print("剩余时间")
                    # print(sleep_time)
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n❌ 运行异常: {e}")
            traceback.print_exc()
        finally:
            # ---- Cleanup ----
            print("\n" + "=" * 60)
            print("🔚 测试结束，清理资源...")
            print("=" * 60)

            # 先关闭夹爪 (夹持力保持 → 夹爪失能)
            try:
                if self.gripper is not None:
                    self.gripper.shutdown()
                    print("[清理] ✅ 夹爪已失能")
            except Exception:
                pass

            # 下使能机器人
            try:
                if self.robot is not None:
                    self.robot.clear_set()
                    self.robot.set_state(arm=self.arm, state=0)
                    self.robot.set_state(arm='B', state=0)
                    self.robot.send_cmd()
                    time.sleep(0.3)
                    print("[清理] ✅ 机器人已下使能")
            except Exception:
                pass

            # 关闭 Space Mouse
            try:
                if self.expert is not None:
                    self.expert.close()
                    print("[清理] ✅ SpaceMouse 已关闭")
            except Exception:
                pass

            print("=" * 60)
            print("✅ 清理完成")
            print("=" * 60)


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == "__main__":
    test = SpaceMouseControlTest()
    test.run()
