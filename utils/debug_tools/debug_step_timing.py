#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marvin 环境 step() 时序分析 - Debug Step Timing

功能：
    完全模拟 marvin_env.py 中 step() 方法的执行过程，详细记录每个
    阶段的耗时信息，用于分析和优化控制频率的性能瓶颈。

主要功能：
    1. 完整模拟 marvin_env.step() 的执行流程
    2. 详细记录每个阶段的时序信息：
       - movLA 轨迹规划耗时
       - 轨迹点发送耗时（总时间和每个点）
       - sleep 等待时间
       - 整个 step 的总耗时
    3. 记录关节状态变化：
       - step 前的关节位置
       - step 后的关节位置
       - 规划的完整轨迹
    4. 数据保存为 JSON 格式供分析和可视化

工作流程：
    1. 初始化：
       - 连接 Marvin 机械臂
       - 初始化运动学引擎（Kine）
       - 配置控制参数（频率、阻抗、movLA参数）

    2. 主控制循环（模拟 step）：
       每个 step 包含以下阶段：

       a) 开始计时：
          - 记录 timestamp_start
          - 读取当前关节位置（joints_before）

       b) movLA 规划阶段：
          - 计算目标关节位置（当前 + delta）
          - 调用 MovLA_InPTPMode 规划轨迹
          - 记录 time_movla（规划耗时）
          - 记录 planned_points（规划的轨迹点）

       c) 轨迹发送阶段：
          - 遍历所有规划的轨迹点
          - 逐点调用 SetArmJoint 发送到机器人
          - 记录每个点的发送时间（send_timestamps）
          - 记录 time_send_total（总发送耗时）

       d) 等待阶段：
          - 计算已用时间（movLA + 发送）
          - sleep 剩余时间以达到控制周期
          - 记录 time_sleep（实际sleep时间）

       e) 结束阶段：
          - 记录 timestamp_end
          - 读取执行后关节位置（joints_after）
          - 计算 total_time（整个step耗时）
          - 检查是否超时（total_time > 控制周期）

    3. 数据保存：
       - 保存完整时序数据到 JSON 文件
       - 文件名包含控制频率和时间戳
       - 保存到 utils/debug_tools/ 目录

数据结构：
    {
        "config": {
            "control_hz": 控制频率,
            "control_period": 控制周期(秒),
            "movla_freq_hz": movLA规划频率,
            "movla_vel": movLA速度,
            "movla_acc": movLA加速度,
            ...
        },
        "steps": [
            {
                "step_id": 步骤编号,
                "timestamp_start": 开始时间戳,
                "timestamp_end": 结束时间戳,
                "joints_before": 执行前关节角度[7],
                "joints_after": 执行后关节角度[7],
                "planned_points": 规划的轨迹点[[7], ...],
                "num_planned_points": 规划点数,
                "time_movla": movLA规划耗时(秒),
                "time_send_total": 总发送耗时(秒),
                "send_timestamps": 每个点的发送时间[],
                "time_sleep": sleep等待时间(秒),
                "total_time": step总耗时(秒),
                "timeout": 是否超时(bool)
            },
            ...
        ]
    }

配置说明：
    - ROBOT_IP: 机械臂 IP 地址
    - ARM: 机械臂编号（'A' 或 'B'）
    - KINE_CONFIG_PATH: 运动学配置文件路径
    - control_hz: 控制频率（通过 --hz 参数指定）
    - movla_freq_hz: movLA 规划频率
    - MOVLA_VEL: movLA 速度参数
    - MOVLA_ACC: movLA 加速度参数

输出：
    - JSON 文件：utils/debug_tools/step_timing_<hz>Hz_<timestamp>.json
    - 控制台：实时显示每个 step 的时序信息

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 测试 10Hz 控制频率
    python utils/debug_tools/debug_step_timing.py --hz 10

    # 测试 20Hz 控制频率
    python utils/debug_tools/debug_step_timing.py --hz 20

    按 Ctrl+C 安全停止

后续分析：
    使用以下工具分析生成的数据：
    1. analyze_trajectory_data.py - 文本统计分析
    2. visualize_trajectory.py - 生成可视化图表

注意事项：
    1. 需要机器人已连接且处于安全状态
    2. 确保控制频率不会导致系统超时
    3. 数据文件会持续累积，注意磁盘空间
    4. 超时的 step 会被标记但不会中断程序
"""

import sys
import os
import time
import signal
import numpy as np
import json
import argparse
from datetime import datetime

# 添加项目路径
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_current_file_dir, '../..'))
serl_infra = os.path.join(project_root, 'serl_robot_infra')
marvin_env_path = os.path.join(serl_infra, 'marvin_env')
sdk_path = os.path.join(marvin_env_path, 'SDK_PYTHON')

sys.path.insert(0, sdk_path)
sys.path.insert(0, marvin_env_path)
sys.path.insert(0, serl_infra)
sys.path.insert(0, project_root)

from fx_robot import Marvin_Robot, DCSS
from fx_kine import Marvin_Kine


# ==============================================================================
# 配置参数
# ==============================================================================

ROBOT_IP = "192.168.14.190"
ARM = 'A'
KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

# Space Mouse 动作缩放
POS_SCALE = 5.0      # mm per control cycle
ROT_SCALE = 0.015    # rad per control cycle
DEADBAND = 0.015

# movLA 参数
MOVLA_VEL = 50
MOVLA_ACC = 50
MOVLA_FREQ_HZ = 100

# 夹爪电机 ID
GRIPPER_MOTOR_ID = 1


# ==============================================================================
# 主测试类
# ==============================================================================

class StepTimingTest:
    """完全模拟 marvin_env.py step() 的测试"""

    def __init__(self, control_hz):
        self.robot = None
        self.kk = None
        self.dcss = None
        self.gripper = None
        self.expert = None

        self.arm = ARM
        self.arm_idx = 0 if ARM == 'A' else 1
        self.control_hz = control_hz
        self.control_period = 1.0 / control_hz

        self.running = False
        self.step_count = 0

        # 数据记录
        self.recorded_steps = []
        self.config_info = {
            "control_hz": control_hz,
            "movla_freq_hz": MOVLA_FREQ_HZ,
            "movla_vel": MOVLA_VEL,
            "movla_acc": MOVLA_ACC,
            "pos_scale": POS_SCALE,
            "rot_scale": ROT_SCALE,
            "timestamp": datetime.now().isoformat(),
        }

    def initialize(self):
        """初始化机器人"""
        print(f"\n[初始化] 连接机器人 {ROBOT_IP}...")

        # 连接机器人
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        ret = self.robot.connect(ROBOT_IP)
        if ret == 0:
            raise RuntimeError(f"Failed to connect to robot at {ROBOT_IP}")
        print("[初始化] ✓ 连接成功")
        time.sleep(0.5)

        # 初始化运动学
        print("[初始化] 加载运动学配置...")
        self.kk = Marvin_Kine()
        self.kk.log_switch(0)

        ini_result = self.kk.load_config(
            arm_type=self.arm_idx,
            config_path=KINE_CONFIG_PATH
        )
        if not ini_result:
            raise RuntimeError("Failed to load kinematics config")

        initial_tag = self.kk.initial_kine(
            robot_type=ini_result['TYPE'][self.arm_idx],
            dh=ini_result['DH'][self.arm_idx],
            pnva=ini_result['PNVA'][self.arm_idx],
            j67=ini_result['BD'][self.arm_idx]
        )
        if not initial_tag:
            raise RuntimeError("Failed to initialize kinematics")
        print("[初始化] ✓ 运动学初始化成功")

        # 初始化 Space Mouse
        print("[初始化] 初始化 Space Mouse...")
        try:
            from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
            self.expert = SpaceMouseExpert()
            print("[初始化] ✓ Space Mouse 初始化成功")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SpaceMouse: {e}")

        # 切换到柔顺模式
        print("[初始化] 切换到柔顺模式...")
        K = np.array([8000.0, 8000.0, 8000.0, 600.0, 600.0, 600.0, 20.0])
        D = np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0])

        self.robot.clear_set()
        self.robot.set_cart_kd_params(arm=self.arm, K=K.tolist(), D=D.tolist(), type=2)
        self.robot.send_cmd()
        time.sleep(0.5)

        self.robot.clear_set()
        self.robot.set_state(arm=self.arm, state=3)
        self.robot.set_impedance_type(arm=self.arm, type=2)
        self.robot.send_cmd()
        time.sleep(0.5)
        print("[初始化] ✓ 已切换到柔顺模式")

        print(f"\n控制频率: {self.control_hz} Hz (周期={self.control_period*1000:.1f}ms)")
        print(f"movLA 规划频率: {MOVLA_FREQ_HZ} Hz")
        print("按 Ctrl+C 保存数据并退出\n")

    def get_current_joints(self):
        """获取当前关节角度"""
        sub = self.robot.subscribe(self.dcss)
        joints_deg = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])
        return joints_deg

    def get_current_pose(self):
        """获取当前位姿 [X,Y,Z,A,B,C]"""
        joints = self.get_current_joints()
        fk_mat = self.kk.fk(joints=joints.tolist())
        xyzabc = self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat)
        return np.array(xyzabc)

    def execute_step(self, action):
        """
        完全模拟 marvin_env.py 的 step() 方法

        流程：
        0. start_time = time.time()
        1. 更新当前位姿
        2a. recover (跳过)
        2b. movLA 规划
        2c. 发送轨迹点
        2d. 模式检查 (跳过)
        3. 夹爪 (跳过)
        4. sleep 到控制周期
        5a. 更新位姿
        5b. get_obs (跳过，直接读关节)
        5c. reward (跳过)
        6. 返回
        """
        t = {}
        step_data = {
            "step_id": self.step_count,
            "action": action.tolist(),
        }

        # ==================== 0. 开始计时 ====================
        t["0_start"] = time.time()

        # ==================== 1. 更新当前位姿 ====================
        joints_before = self.get_current_joints()
        current_xyzabc = self.get_current_pose()
        t["1_update_currpos"] = time.time()

        step_data["joints_before"] = joints_before.tolist()
        step_data["pose_before"] = current_xyzabc.tolist()

        # ==================== 2a. recover (跳过) ====================
        t["2a_recover"] = time.time()

        # ==================== 2b. movLA 规划 ====================
        # 计算目标位姿
        pos_delta_mm = action[:3] * POS_SCALE
        rot_delta_rad = action[3:6] * ROT_SCALE

        target_xyzabc = current_xyzabc.copy()
        target_xyzabc[:3] += pos_delta_mm
        target_xyzabc[3:] += np.rad2deg(rot_delta_rad)

        # movLA 规划
        delta_dist_mm = np.linalg.norm(pos_delta_mm)
        delta_rot_rad = np.linalg.norm(rot_delta_rad)

        if delta_dist_mm > 0.01 or delta_rot_rad > 1e-5:
            points, _ = self.kk.movLA(
                start_xyzabc=current_xyzabc.tolist(),
                end_xyzabc=target_xyzabc.tolist(),
                ref_joints=joints_before.tolist(),
                vel=MOVLA_VEL,
                acc=MOVLA_ACC,
                freq_hz=MOVLA_FREQ_HZ
            )
        else:
            points = None

        t["2b_movLA"] = time.time()

        # ==================== 2c. 发送轨迹点 ====================
        if points and len(points) > 0:
            original_count = len(points)

            # 🔧 压缩（和 marvin_env.py 一样）
            if original_count >= 2:
                compressed_points = [points[i] for i in range(0, original_count, 2)]
                if (original_count - 1) % 2 != 0:
                    compressed_points.append(points[-1])
                points = compressed_points

            # 记录发送时间戳
            send_timestamps = []
            for i, pt in enumerate(points):
                t_pt_start = time.time()
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                self.robot.send_cmd()
                t_pt_end = time.time()
                send_timestamps.append(t_pt_end - t_pt_start)

            step_data["num_planned_points"] = original_count
            step_data["num_sent_points"] = len(points)
            step_data["planned_points"] = points  # 所有发送的点
            step_data["send_timestamps"] = send_timestamps
        else:
            step_data["num_planned_points"] = 0
            step_data["num_sent_points"] = 0

        t["2c_send_joints"] = time.time()

        # ==================== 2d. 模式检查 (跳过) ====================
        t["2d_mode_check"] = time.time()

        # ==================== 3. 夹爪 (跳过) ====================
        t["3_gripper"] = time.time()

        # ==================== 4. 频率控制 (sleep) ====================
        dt = time.time() - t["0_start"]
        sleep_time = max(0, self.control_period - dt - 0.001)  # 留 1ms 余量
        if sleep_time > 0:
            time.sleep(sleep_time)
        t["4_sleep"] = time.time()

        # ==================== 5a. 更新位姿 ====================
        joints_after = self.get_current_joints()
        pose_after = self.get_current_pose()
        t["5a_update_currpos"] = time.time()

        step_data["joints_after"] = joints_after.tolist()
        step_data["pose_after"] = pose_after.tolist()

        # ==================== 5b. get_obs (跳过) ====================
        t["5b_get_obs"] = time.time()

        # ==================== 5c. reward (跳过) ====================
        t["5c_reward"] = time.time()

        # ==================== 6. 返回 ====================
        t["6_return"] = time.time()

        # 计算各阶段耗时
        step_data["timestamp_start"] = t["0_start"]
        step_data["timestamp_end"] = t["6_return"]
        step_data["time_update1"] = t["1_update_currpos"] - t["0_start"]
        step_data["time_recover"] = t["2a_recover"] - t["1_update_currpos"]
        step_data["time_movla"] = t["2b_movLA"] - t["2a_recover"]
        step_data["time_send_total"] = t["2c_send_joints"] - t["2b_movLA"]
        step_data["time_mode_check"] = t["2d_mode_check"] - t["2c_send_joints"]
        step_data["time_gripper"] = t["3_gripper"] - t["2d_mode_check"]
        step_data["time_sleep"] = t["4_sleep"] - t["3_gripper"]
        step_data["time_update2"] = t["5a_update_currpos"] - t["4_sleep"]
        step_data["time_obs"] = t["5b_get_obs"] - t["5a_update_currpos"]
        step_data["time_reward"] = t["5c_reward"] - t["5b_get_obs"]
        step_data["time_final"] = t["6_return"] - t["5c_reward"]
        step_data["total_time"] = t["6_return"] - t["0_start"]

        # 打印（和 marvin_env.py 一样）
        dt_total = step_data["total_time"] * 1000
        dt_update1 = step_data["time_update1"] * 1000
        dt_movla = step_data["time_movla"] * 1000
        dt_send = step_data["time_send_total"] * 1000
        dt_sleep = step_data["time_sleep"] * 1000
        dt_update2 = step_data["time_update2"] * 1000

        print(f"[step={self.step_count:3d}][TIMING] total={dt_total:.1f}ms | "
              f"update1={dt_update1:.1f} movLA={dt_movla:.1f} send={dt_send:.1f} "
              f"sleep={dt_sleep:.1f} update2={dt_update2:.1f} | "
              f"points={step_data.get('num_sent_points', 0)}")

        self.recorded_steps.append(step_data)
        self.step_count += 1

    def run(self):
        """主控制循环"""
        self.running = True
        print("[开始] 进入控制循环，移动 Space Mouse...\n")

        try:
            while self.running:
                # 获取 Space Mouse 输入
                raw_data = self.expert.get_action()
                if isinstance(raw_data, tuple):
                    action, buttons = raw_data
                else:
                    action = raw_data
                    buttons = [0, 0]

                # 转换为 numpy 数组
                try:
                    action = np.array(action, dtype=np.float64)
                except:
                    continue

                # 检查死区
                if action.shape != (6,):
                    continue

                action_norm = np.linalg.norm(action)
                if action_norm < DEADBAND:
                    continue

                # 执行 step
                action_7d = np.append(action, 0.0)  # 添加夹爪动作 (0=不动)
                self.execute_step(action_7d)

        except KeyboardInterrupt:
            print("\n\n[信号] 收到 Ctrl+C，正在保存数据...")
            self.running = False

    def save_data(self):
        """保存数据到 JSON 文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"step_timing_{self.control_hz}hz_{timestamp}.json"

        # 保存到 utils/debug_tools 目录下
        output_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(output_dir, filename)

        data = {
            "config": self.config_info,
            "steps": self.recorded_steps,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[保存] 数据已保存到: {filepath}")
        print(f"[保存] 共记录 {len(self.recorded_steps)} 个 step")

    def shutdown(self):
        """关闭并清理"""
        print("\n[关闭] 正在关闭...")
        if self.robot:
            print("[关闭] 释放机器人连接...")
            self.robot.release()
        print("[关闭] 完成")


# ==============================================================================
# 主程序
# ==============================================================================

def signal_handler(sig, frame):
    """Ctrl+C 信号处理"""
    print("\n[信号] 收到 Ctrl+C...")


def main():
    parser = argparse.ArgumentParser(description='Step 执行时间记录测试')
    parser.add_argument('--hz', type=int, required=True, choices=[10, 20],
                        help='控制频率 (10 或 20 Hz)')
    args = parser.parse_args()

    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)

    test = StepTimingTest(control_hz=args.hz)

    try:
        test.initialize()
        test.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if test.step_count > 0:
            test.save_data()
        test.shutdown()


if __name__ == "__main__":
    main()
