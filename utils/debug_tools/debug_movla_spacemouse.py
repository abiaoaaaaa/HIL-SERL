#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Space Mouse 控制测试与数据记录 - Debug movLA Space Mouse Control

功能：
    模拟或真实执行 Space Mouse 控制机械臂，详细记录每个控制周期的
    规划轨迹和实际执行情况，用于分析控制频率对轨迹平滑度的影响。

主要功能：
    1. 支持不同控制频率测试（10Hz、20Hz等）
    2. 记录每个控制周期的详细数据：
       - movLA 规划的完整轨迹点（planned_points）
       - 实际执行后的关节位置（joints_after）
       - 各阶段耗时（规划时间、发送时间、总时间）
    3. 对比分析：实际执行到了规划轨迹的第几个点
    4. 数据保存为 JSON 格式供后续可视化和分析

工作流程：
    1. 初始化：
       - 连接 Marvin 机械臂
       - 初始化运动学引擎（Kine）
       - 配置控制参数（频率、阻抗等）
       - 可选：初始化 Space Mouse

    2. 主控制循环：
       对每个控制周期：
       a) 读取当前关节位置（joints_before）
       b) 生成或读取目标增量（delta）
       c) movLA 规划轨迹：
          - 调用 MovLA_InPTPMode 生成轨迹点序列
          - 记录规划时间（time_movla）
       d) 发送轨迹到机器人：
          - 逐点发送 SetArmJoint
          - 记录每个点的发送时间
       e) 等待执行：
          - sleep 至控制周期结束
       f) 读取执行后位置（joints_after）
       g) 分析执行情况：
          - 对比 joints_after 与规划的每个点
          - 找出实际执行到了第几个规划点

    3. 数据保存：
       - 保存完整数据到 JSON 文件
       - 文件名包含控制频率和时间戳
       - 保存到 utils/debug_tools/ 目录

数据结构：
    {
        "config": {
            "control_hz": 控制频率,
            "movla_freq_hz": movLA规划频率,
            ...
        },
        "steps": [
            {
                "step_id": 步骤编号,
                "timestamp_start": 开始时间戳,
                "joints_before": 执行前关节角度[7],
                "joints_after": 执行后关节角度[7],
                "planned_points": 规划的轨迹点[[7], ...],
                "num_planned_points": 规划点数,
                "time_movla": movLA规划耗时(秒),
                "time_send_total": 发送总耗时(秒),
                "send_timestamps": 每个点的发送时间[],
                "total_time": 总耗时(秒),
                "executed_point_index": 执行到第几个点
            },
            ...
        ]
    }

配置说明：
    - ROBOT_IP: 机械臂 IP 地址
    - ARM: 机械臂编号（'A' 或 'B'）
    - KINE_CONFIG_PATH: 运动学配置文件路径
    - control_hz: 控制频率（通过 --hz 参数指定）
    - movla_freq_hz: movLA 规划频率（默认 500Hz）

输出：
    - JSON 文件：utils/debug_tools/trajectory_data_<hz>Hz_<timestamp>.json
    - 控制台：实时显示每个周期的执行情况

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 模拟模式（不连接真实机器人）
    python utils/debug_tools/debug_movla_spacemouse.py --hz 10 --simulate
    python utils/debug_tools/debug_movla_spacemouse.py --hz 20 --simulate

    # 真实模式（连接真实机器人和Space Mouse）
    python utils/debug_tools/debug_movla_spacemouse.py --hz 10
    python utils/debug_tools/debug_movla_spacemouse.py --hz 20

    按 Ctrl+C 安全停止

注意事项：
    1. 真实模式需要机器人已连接且处于安全状态
    2. Space Mouse 需要正确连接
    3. 数据文件会持续累积，注意磁盘空间
    4. 分析数据使用 analyze_trajectory_data.py 或 visualize_trajectory.py
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

POS_SCALE = 20.0      # mm per control cycle
ROT_SCALE = 0.015    # rad per control cycle
DEADBAND = 0.015

MOVLA_FREQ_HZ = 100
MOVLA_VEL = 100
MOVLA_ACC = 100

SAFETY_LOW = np.array([342.2, 258.1, 109.1, -109.1, -16.3, -103.2])
SAFETY_HIGH = np.array([516.5, 390.1, 381.9, -77.5, 23.5, -71.3])

GRIPPER_MOTOR_ID = 1

# 全局数据记录
trajectory_data = {
    "config": {},
    "steps": []
}


# ==============================================================================
# 工具函数
# ==============================================================================

def save_trajectory_data(filename):
    """保存轨迹数据到 JSON 文件"""
    # 保存到 utils/debug_tools 目录下
    output_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    print(f"\n[保存] 数据已保存到: {filepath}")
    print(f"[保存] 总共记录 {len(trajectory_data['steps'])} 个 step")


# ==============================================================================
# 主测试类
# ==============================================================================

class DebugSpaceMouseTest:
    """简化版 Space Mouse 控制测试"""

    def __init__(self, control_hz):
        self.robot = None
        self.kk = None
        self.dcss = None
        self.gripper = None
        self.expert = None

        self.arm = ARM
        self.arm_idx = 0 if ARM == 'A' else 1
        self.control_hz = control_hz

        self.running = False

        # 更新配置
        trajectory_data["config"] = {
            "control_hz": control_hz,
            "movla_freq_hz": MOVLA_FREQ_HZ,
            "movla_vel": MOVLA_VEL,
            "movla_acc": MOVLA_ACC,
            "pos_scale": POS_SCALE,
            "rot_scale": ROT_SCALE,
            "timestamp": datetime.now().isoformat()
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

        # 初始化夹爪
        print("[初始化] 初始化夹爪...")
        try:
            from marvin_env.gripper.marvin_gripper import MarvinGripperController
            self.gripper = MarvinGripperController(
                robot=self.robot,
                arm=self.arm,
                motor_id=GRIPPER_MOTOR_ID
            )
            if self.gripper.initialize():
                print("[初始化] ✓ 夹爪初始化成功")
            else:
                print("[初始化] ✗ 夹爪初始化失败")
                self.gripper = None
        except Exception as e:
            print(f"[初始化] ✗ 夹爪初始化失败: {e}")
            self.gripper = None

        # 🔧 模拟模式下不需要初始化 Space Mouse
        if not hasattr(self, 'simulate_mode'):
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

        print("\n" + "="*70)
        print(f"数据记录模式:")
        print(f"  控制频率: {self.control_hz} Hz")
        print(f"  movLA freq_hz: {MOVLA_FREQ_HZ} Hz")
        print(f"  无 sleep（瞬间发送所有点）")
        print(f"  动作缩放: pos={POS_SCALE}mm, rot={ROT_SCALE}rad")
        print("="*70)
        if hasattr(self, 'simulate_mode'):
            print("模式: 模拟执行 3 个固定周期")
        else:
            print("控制:")
            print("  移动 Space Mouse 控制机械臂")
            print("  Ctrl+C: 保存数据并退出")
        print("="*70 + "\n")

    def get_current_pose(self):
        """获取当前位姿 [X,Y,Z,A,B,C] (mm, deg)"""
        sub = self.robot.subscribe(self.dcss)
        joints_deg = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])

        fk_mat = self.kk.fk(joints=joints_deg.tolist())
        current_xyzabc = self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat)

        return np.array(current_xyzabc)

    def safe_move(self, pos_delta_mm, rot_delta_rad):
        """执行安全移动并记录数据"""
        step_start = time.time()

        step_data = {
            "step_id": len(trajectory_data["steps"]) + 1,
            "timestamp_start": step_start,
            "space_mouse_delta": {
                "pos_mm": pos_delta_mm.tolist(),
                "rot_rad": rot_delta_rad.tolist()
            }
        }

        # 1. 获取移动前的关节角度
        t_before_joints = time.time()
        current = self.get_current_pose()
        sub = self.robot.subscribe(self.dcss)
        current_joints_before = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])
        step_data["joints_before"] = current_joints_before.tolist()
        step_data["pose_before"] = current.tolist()
        step_data["time_read_joints_before"] = time.time() - t_before_joints

        # 计算目标位姿
        target = current.copy()
        target[:3] += pos_delta_mm
        target[3:] += np.rad2deg(rot_delta_rad)
        step_data["target_pose"] = target.tolist()

        # 安全检查
        if not (np.all(target[:3] >= SAFETY_LOW[:3]) and
                np.all(target[:3] <= SAFETY_HIGH[:3])):
            print(f"[警告] 目标位置超出安全范围，跳过")
            step_data["skipped"] = True
            step_data["skip_reason"] = "out_of_bounds"
            trajectory_data["steps"].append(step_data)
            return

        # 2. movLA 规划
        t_movla_start = time.time()
        try:
            points, pset = self.kk.movLA(
                start_xyzabc=current.tolist(),
                end_xyzabc=target.tolist(),
                ref_joints=current_joints_before.tolist(),
                vel=MOVLA_VEL,
                acc=MOVLA_ACC,
                freq_hz=MOVLA_FREQ_HZ
            )
        except Exception as e:
            print(f"[错误] movLA 规划失败: {e}")
            step_data["skipped"] = True
            step_data["skip_reason"] = f"movla_failed: {e}"
            trajectory_data["steps"].append(step_data)
            return

        t_movla = time.time() - t_movla_start
        step_data["time_movla"] = t_movla

        if not points or len(points) == 0:
            print(f"[警告] movLA 返回空轨迹")
            step_data["skipped"] = True
            step_data["skip_reason"] = "empty_trajectory"
            trajectory_data["steps"].append(step_data)
            return

        # 记录规划的轨迹点
        step_data["planned_points"] = [pt for pt in points]
        step_data["num_planned_points"] = len(points)

        print(f"[safe_move] 规划 {len(points)} 点，瞬间发送所有点，然后 sleep 50ms")

        # 3. 瞬间发送所有轨迹点（不加 sleep）
        t_send_start = time.time()
        send_timestamps = []

        for i, pt in enumerate(points):
            t_point_start = time.time()
            self.robot.clear_set()
            self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
            self.robot.send_cmd()
            send_timestamps.append(time.time() - t_point_start)

        t_send = time.time() - t_send_start
        step_data["time_send_total"] = t_send
        step_data["send_timestamps"] = send_timestamps
        step_data["num_sent_points"] = len(points)

        # 4. 固定 sleep 50ms
        time.sleep(0.05)
        step_data["time_sleep_total"] = 0.05

        # 5. 获取移动后的关节角度（在 sleep 后读取）
        t_after_joints = time.time()
        sub_after = self.robot.subscribe(self.dcss)
        current_joints_after = np.array(sub_after['outputs'][self.arm_idx]['fb_joint_pos'])
        current_pose_after = self.get_current_pose()
        step_data["joints_after"] = current_joints_after.tolist()
        step_data["pose_after"] = current_pose_after.tolist()
        step_data["time_read_joints_after"] = time.time() - t_after_joints

        # 计算实际移动距离
        actual_delta_mm = np.linalg.norm(current_pose_after[:3] - current[:3])
        step_data["actual_delta_mm"] = actual_delta_mm

        step_data["timestamp_end"] = time.time()
        step_data["total_time"] = step_data["timestamp_end"] - step_data["timestamp_start"]

        # 保存数据
        trajectory_data["steps"].append(step_data)

        # 打印简要信息
        print(f"[step {step_data['step_id']:4d}] 规划={len(points):3d}点 | 发送={t_send*1000:5.1f}ms | "
              f"sleep=50ms | Δpos={actual_delta_mm:.1f}mm")

    def run_simulation(self):
        """模拟执行 3 个控制周期"""
        print("\n[模拟模式] 执行 3 个控制周期...")
        print(f"[模拟模式] 控制频率: {self.control_hz} Hz\n")

        # 模拟的 Space Mouse 输入（固定动作）
        simulated_actions = [
            {"pos_mm": np.array([5.0, 3.0, 0.0]), "rot_rad": np.array([0.0, 0.0, 0.01])},
            {"pos_mm": np.array([4.0, -2.0, 1.0]), "rot_rad": np.array([0.005, 0.0, -0.005])},
            {"pos_mm": np.array([-3.0, 5.0, -1.0]), "rot_rad": np.array([0.0, 0.01, 0.0])},
        ]

        for i, action in enumerate(simulated_actions):
            print(f"\n{'='*70}")
            print(f"[模拟周期 {i+1}/3] 开始执行")
            print(f"输入: pos={action['pos_mm']} mm, rot={action['rot_rad']} rad")
            print(f"{'='*70}\n")

            pos_delta_mm = action['pos_mm']
            rot_delta_rad = action['rot_rad']

            # 🔧 如果是 10Hz，内部分两次执行
            if self.control_hz == 10:
                print(f"[10Hz 模式] 分两次执行（每次 50ms）")

                # 第一次：执行前半段
                print(f"\n  [Sub-step 1/2] 执行前半段...")
                self.safe_move(pos_delta_mm / 2, rot_delta_rad / 2)

                # 第二次：执行后半段
                print(f"\n  [Sub-step 2/2] 执行后半段...")
                self.safe_move(pos_delta_mm / 2, rot_delta_rad / 2)
            else:
                # 20Hz：直接执行完整动作
                print(f"[20Hz 模式] 一次执行完整动作")
                self.safe_move(pos_delta_mm, rot_delta_rad)

            print(f"\n{'='*70}")
            print(f"[模拟周期 {i+1}/3] 完成")
            print(f"{'='*70}\n")

        print("\n[模拟模式] 所有周期执行完成！")
        self.analyze_simulation()

    def analyze_simulation(self):
        """分析模拟数据：对比每个 step 的规划点和实际执行位置"""
        print("\n" + "="*70)
        print("📊 模拟数据分析")
        print("="*70 + "\n")

        for step_idx, step in enumerate(trajectory_data["steps"]):
            print(f"\n--- Step {step['step_id']} ---")

            # 规划信息
            planned_points = np.array(step['planned_points'])
            num_planned = len(planned_points)
            print(f"规划点数: {num_planned}")

            # 执行前后的关节位置
            joints_before = np.array(step['joints_before'])
            joints_after = np.array(step['joints_after'])

            # 计算实际执行到了第几个点
            # 方法：找到与 joints_after 最接近的规划点
            distances = np.linalg.norm(planned_points - joints_after, axis=1)
            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]

            print(f"实际执行到: 第 {closest_idx+1}/{num_planned} 个点")
            print(f"与最近规划点的距离: {min_distance:.4f}°")
            print(f"执行进度: {(closest_idx+1)/num_planned*100:.1f}%")

            # 分析是否执行完整
            if closest_idx < num_planned - 5:
                print(f"⚠️ 警告: 轨迹未执行完整！只执行了 {(closest_idx+1)/num_planned*100:.1f}%")
            else:
                print(f"✅ 轨迹基本执行完整")

            # 打印时间信息
            print(f"movLA 规划耗时: {step['time_movla']*1000:.2f}ms")
            print(f"发送轨迹耗时: {step['time_send_total']*1000:.2f}ms")
            print(f"sleep 时间: {step['time_sleep_total']*1000:.2f}ms")

    def run(self):
        """主控制循环 - 折衷方案：10Hz 控制，内部分两次 50ms 执行"""
        self.running = True
        print("\n[开始] 进入控制循环...")
        print("[提示] 移动 Space Mouse 开始记录数据，按 Ctrl+C 保存并退出\n")

        control_period = 1.0 / self.control_hz

        while self.running:
            loop_start = time.time()

            try:
                # 读取 Space Mouse 输入
                action, buttons = self.expert.get_action()

                if action is None:
                    time.sleep(0.01)
                    continue

                action = np.array(action, dtype=np.float64)

                # 检查死区
                if np.linalg.norm(action) < DEADBAND:
                    elapsed = time.time() - loop_start
                    if elapsed < control_period:
                        time.sleep(control_period - elapsed)
                    continue

                # 解析动作
                pos = action[:3]
                rot = action[3:]

                # Space Mouse 坐标映射
                pos_delta_mm = np.array([
                    -pos[1] * POS_SCALE,
                    pos[0] * POS_SCALE,
                    pos[2] * POS_SCALE
                ])
                rot_delta_rad = np.array([
                    -rot[0] * ROT_SCALE,
                    -rot[1] * ROT_SCALE,
                    -rot[2] * ROT_SCALE
                ])

                # 🔧 折衷方案：如果是 10Hz，内部分两次执行（每次 50ms）
                if self.control_hz == 10:
                    # 第一次：执行一半动作
                    print(f"\n[10Hz->20Hz] Step 1/2: 执行前半段...")
                    self.safe_move(pos_delta_mm / 2, rot_delta_rad / 2)

                    # 第二次：执行剩余一半动作
                    print(f"[10Hz->20Hz] Step 2/2: 执行后半段...")
                    self.safe_move(pos_delta_mm / 2, rot_delta_rad / 2)
                else:
                    # 20Hz：直接执行完整动作
                    self.safe_move(pos_delta_mm, rot_delta_rad)

            except Exception as e:
                print(f"[错误] 控制循环异常: {e}")
                import traceback
                traceback.print_exc()

            # 注意：sleep 已在 safe_move() 内部处理，这里不需要额外 sleep

    def shutdown(self):
        """关闭"""
        print("\n[关闭] 正在关闭...")
        self.running = False

        if self.gripper:
            print("[关闭] 关闭夹爪...")
            try:
                self.gripper.shutdown()
            except:
                pass

        if self.robot:
            print("[关闭] 释放机器人连接...")
            try:
                self.robot.release_robot()
            except:
                pass

        print("[关闭] 完成")


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Space Mouse 数据记录测试')
    parser.add_argument('--hz', type=int, choices=[10, 20], required=True,
                        help='控制频率 (10 或 20 Hz)')
    parser.add_argument('--simulate', action='store_true',
                        help='模拟模式：执行 3 个固定周期并分析')
    args = parser.parse_args()

    test = DebugSpaceMouseTest(control_hz=args.hz)

    # 🔧 设置模拟模式标志
    if args.simulate:
        test.simulate_mode = True

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n[信号] 收到 Ctrl+C，正在退出...")
        test.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        test.initialize()

        # 🔧 根据模式选择运行方式
        if args.simulate:
            test.run_simulation()  # 模拟模式：执行 3 个周期
        else:
            test.run()  # 正常模式：Space Mouse 控制

    except KeyboardInterrupt:
        print("\n[中断] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.shutdown()

        # 保存数据
        if args.simulate:
            filename = f"simulation_{args.hz}hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"trajectory_data_{args.hz}hz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_trajectory_data(filename)


if __name__ == "__main__":
    main()
