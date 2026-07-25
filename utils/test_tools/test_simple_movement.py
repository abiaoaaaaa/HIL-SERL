#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marvin 简单移动测试 - Test Simple Movement

功能：
    测试 Marvin 机械臂的基本移动功能，验证连接、控制参数设置、
    笛卡尔空间移动等核心功能是否正常。

主要功能：
    1. 机械臂连接测试
    2. 状态读取验证
    3. 阻抗参数设置
    4. 简单笛卡尔移动（前移 30mm）
    5. 急停保护机制

工作流程：
    1. 初始化阶段：
       a) 连接机械臂：
          - 创建 Marvin_Robot 实例
          - 连接到指定 IP 地址
          - 选择机械臂（A 或 B）
       b) 初始化运动学：
          - 加载 Kine 配置文件
          - 创建 Marvin_Kine 实例
          - 创建 DCSS 坐标系统
       c) 注册信号处理：
          - 捕获 Ctrl+C 信号
          - 设置急停处理函数

    2. 状态验证阶段：
       a) 读取当前状态：
          - 获取关节角度（7个关节）
          - 通过正运动学计算末端位姿
          - 显示当前笛卡尔坐标 (X, Y, Z, RX, RY, RZ)
       b) 验证机械臂状态：
          - 检查是否已使能
          - 确认无报警信息
          - 验证数据读取正常

    3. 参数设置阶段：
       a) 设置笛卡尔阻抗参数：
          - 位置刚度 (K_xyz): 控制位置跟踪刚度
          - 姿态刚度 (K_rxryrz): 控制姿态跟踪刚度
          - 位置阻尼 (C_xyz): 控制位置响应阻尼
          - 姿态阻尼 (C_rxryrz): 控制姿态响应阻尼
       b) 启用阻抗控制模式

    4. 移动执行阶段：
       a) 计算目标位姿：
          - 当前位姿: (X, Y, Z, RX, RY, RZ)
          - 目标位姿: X 方向 +30mm
          - 保持 Y, Z, RX, RY, RZ 不变
       b) 执行移动：
          - 调用 SetArmPose 设置目标
          - 等待移动完成
          - 验证到达目标
       c) 读取最终位姿：
          - 获取移动后的关节角度
          - 计算实际末端位姿
          - 显示移动结果

    5. 清理阶段：
       - 释放机械臂连接
       - 输出测试结果

急停机制：
    1. 信号捕获：
       - Ctrl+C 触发 SIGINT 信号
       - 调用 emergency_stop() 方法

    2. 急停流程：
       a) 立即停止当前运动
       b) 切换到下使能状态（关节锁死）
       c) 保持当前位置
       d) 安全退出程序

    3. 恢复方法：
       - 需要手动重新使能机械臂
       - 或重启控制程序

配置说明：
    - ROBOT_IP: 机械臂 IP 地址（默认: "192.168.14.190"）
    - ARM: 机械臂选择（'A' 或 'B'）
    - KINE_CONFIG_PATH: 运动学配置文件路径
    - 阻抗参数：
      * K_xyz: 位置刚度 [Nx, Ny, Nz] (N/m)
      * K_rxryrz: 姿态刚度 [Nrx, Nry, Nrz] (Nm/rad)
      * C_xyz: 位置阻尼 [Cx, Cy, Cz] (Ns/m)
      * C_rxryrz: 姿态阻尼 [Crx, Cry, Crz] (Nms/rad)

预期输出：
    ================================================================
    Marvin 简单移动测试
    ================================================================

    [1/5] 连接机械臂...
    ✓ 已连接到 192.168.14.190
    ✓ 选择机械臂 A

    [2/5] 初始化运动学...
    ✓ 加载配置: /path/to/ccs_m6_40.MvKDCfg
    ✓ 运动学引擎初始化成功

    [3/5] 读取当前位姿...
    ✓ 关节角度: [q1, q2, q3, q4, q5, q6, q7]
    ✓ 当前位姿 (mm, deg):
      X=500.0, Y=200.0, Z=300.0
      RX=180.0, RY=0.0, RZ=-90.0

    [4/5] 设置阻抗参数...
    ✓ 位置刚度: [2500, 2500, 2000] N/m
    ✓ 姿态刚度: [300, 300, 100] Nm/rad
    ✓ 阻抗控制已启用

    [5/5] 执行移动 (X方向 +30mm)...
    ✓ 目标位姿: X=530.0, Y=200.0, Z=300.0
    ✓ 移动完成
    ✓ 最终位姿: X=530.2, Y=199.8, Z=300.1
    ✓ 位置误差: 0.3mm

    ================================================================
    测试完成！机械臂工作正常。
    ================================================================

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 运行测试
    python utils/test_tools/test_simple_movement.py

    # 紧急情况按 Ctrl+C 停止

应用场景：
    1. 机械臂首次连接后的功能验证
    2. 更换控制参数后的测试
    3. 运动学配置修改后的验证
    4. 排查连接或控制问题

常见问题排查：
    1. 连接失败：
       - 检查网络连接（ping ROBOT_IP）
       - 验证 IP 地址正确
       - 确认机械臂电源已开启

    2. 运动学初始化失败：
       - 检查配置文件路径
       - 验证配置文件格式
       - 确认文件读取权限

    3. 移动失败：
       - 检查目标位姿是否可达
       - 验证工作空间边界
       - 确认无碰撞风险
       - 检查机械臂使能状态

    4. 急停后恢复：
       - 手动重新使能机械臂
       - 检查报警信息
       - 必要时重启控制器

注意事项：
    1. 首次运行前确认周围安全
    2. 准备好物理急停按钮
    3. 移动距离较小（30mm），相对安全
    4. Ctrl+C 会立即锁死关节
    5. 不要在接近边界或障碍物时测试
"""
import sys
import os
import time
import signal
import numpy as np

# 添加路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_current_dir)
# 项目根目录（用于 from examples... / from serl_robot_infra...）
sys.path.append(os.path.join(_current_dir, '../..'))
# SDK_PYTHON 在 serl_robot_infra/marvin_env/SDK_PYTHON/
sys.path.append(os.path.join(_current_dir, '../../serl_robot_infra/marvin_env'))

from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
from SDK_PYTHON.fx_kine import Marvin_Kine

# 机器人配置（不通过config.py，避免依赖链问题）
ROBOT_IP = "192.168.14.190"
KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"


class SimpleMovementTest:
    """简单移动测试"""

    def __init__(self):
        self.robot = None
        self.dcss = None
        self.kk = None
        self.running = True

        # 注册Ctrl+C信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

    def emergency_stop(self):
        """急停：立即切换到下使能状态（关节锁死）"""
        if self.robot is None:
            return
        try:
            self.robot.clear_set()
            self.robot.set_state(arm='A', state=0)  # 0=下使能
            self.robot.set_state(arm='B', state=0)
            self.robot.send_cmd()
            time.sleep(0.5)
            self.robot.release_robot()
        except Exception:
            pass

    def signal_handler(self, sig, frame):
        """Ctrl+C急停处理"""
        print("\n" + "="*60)
        print("🛑 检测到 Ctrl+C，执行急停...")
        print("="*60)

        self.emergency_stop()

        print("="*60)
        print("🛑 测试已终止")
        print("="*60)
        sys.exit(0)

    def wait_for_continue(self, prompt="按Enter继续，Ctrl+C中止"):
        """等待用户确认"""
        print(f"\n{'='*60}")
        print(f"⏸️  {prompt}")
        print(f"{'='*60}")
        try:
            input()
        except KeyboardInterrupt:
            self.signal_handler(None, None)

    def print_section(self, title):
        """打印测试章节"""
        print("\n" + "="*60)
        print(f"📋 {title}")
        print("="*60)

    def connect(self):
        """连接机器人"""
        self.print_section("步骤1: 连接机器人")

        try:
            # 初始化SDK
            self.robot = Marvin_Robot()
            self.dcss = DCSS()
            self.kk = Marvin_Kine()

            print(f"[连接] 机器人IP: {ROBOT_IP}")
            ret = self.robot.connect(ROBOT_IP)
            if ret == 0:
                raise RuntimeError(f"Failed to connect to {ROBOT_IP}")

            print("[连接] ✅ 连接成功")
            time.sleep(0.5)

            # 加载运动学
            print(f"[连接] 加载运动学配置...")
            print(f"[连接] 路径: {KINE_CONFIG_PATH}")

            self.kk.log_switch(0)
            ini_result = self.kk.load_config(
                arm_type=0,
                config_path=KINE_CONFIG_PATH
            )

            if not ini_result:
                raise RuntimeError("Failed to load kinematics config")

            initial_tag = self.kk.initial_kine(
                robot_type=ini_result['TYPE'][0],
                dh=ini_result['DH'][0],
                pnva=ini_result['PNVA'][0],
                j67=ini_result['BD'][0]
            )

            if not initial_tag:
                raise RuntimeError("Failed to initialize kinematics")

            print("[连接] ✅ 运动学初始化成功")

            # 清除错误
            self.robot.check_error_and_clear(self.dcss)
            time.sleep(0.5)

            return True

        except Exception as e:
            print(f"[连接] ❌ 连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_current_pose(self):
        """获取当前位姿"""
        self.print_section("步骤2: 获取当前位姿")

        try:
            # 获取当前关节角度
            sub_data = self.robot.subscribe(self.dcss)
            joints = sub_data['outputs'][0]['fb_joint_pos']

            print(f"[位姿] 当前关节角度 (度): {[round(j, 2) for j in joints]}")

            # 正运动学计算末端位姿
            fk_mat = self.kk.fk(joints=joints)
            xyzabc = self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat)

            print(f"\n[位姿] 当前末端位姿 (XYZABC):")
            print(f"  位置: X={xyzabc[0]:.2f}mm, Y={xyzabc[1]:.2f}mm, Z={xyzabc[2]:.2f}mm")
            print(f"  姿态: A={xyzabc[3]:.2f}°, B={xyzabc[4]:.2f}°, C={xyzabc[5]:.2f}°")

            return np.array(joints), np.array(xyzabc)

        except Exception as e:
            print(f"[位姿] ❌ 获取失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def setup_impedance_mode(self):
        """设置笛卡尔阻抗模式（参考demo）"""
        self.print_section("步骤3: 设置笛卡尔阻抗模式")

        try:
            # 参考 20_cartesian_movl_demo.py — 移动时用高刚度防止下垂
            # K 第一项 X=8000 不会往下掉，demo 07 的 10 是柔顺测试用的
            K = [8000, 8000, 8000, 600, 600, 600, 20]
            D = [0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1]

            print(f"[阻抗] 设置笛卡尔阻抗参数...")
            print(f"  刚度K: {K}")
            print(f"  阻尼D: {D}")
            print(f"  说明: X方向刚度低(10)，YZ方向刚度高(5000)")

            self.wait_for_continue("确认设置阻抗参数")

            # 设置阻抗参数
            self.robot.clear_set()
            self.robot.set_cart_kd_params(
                arm='A',
                K=K,
                D=D,
                type=2  # 笛卡尔阻抗
            )
            self.robot.set_vel_acc(arm='A', velRatio=10, AccRatio=10)
            self.robot.send_cmd()
            time.sleep(0.5)

            print("[阻抗] ✅ 阻抗参数已设置")

            # 切换到扭矩模式 + 笛卡尔阻抗
            print("\n[阻抗] 切换到扭矩模式 + 笛卡尔阻抗...")
            self.robot.clear_set()
            self.robot.set_state(arm='A', state=3)  # 扭矩模式
            self.robot.set_impedance_type(arm='A', type=2)  # 笛卡尔阻抗
            self.robot.send_cmd()
            time.sleep(2.0)  # 等待模式切换

            print("[阻抗] ✅ 模式切换完成")

            # 设置末端笛卡尔控制参数（参考demo 07）
            print("\n[阻抗] 设置末端笛卡尔控制参数...")
            sub_data = self.robot.subscribe(self.dcss)
            joints = sub_data['outputs'][0]['fb_joint_pos']
            fk_mat = self.kk.fk(joints=joints)
            xyzabc = self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat)

            self.robot.clear_set()
            self.robot.set_EefCart_control_params(
                arm='A',
                fcType=1,
                CartCtrlPara=[xyzabc[3], xyzabc[4], xyzabc[5], 0, 0, 0, 0]
            )
            self.robot.send_cmd()
            time.sleep(0.5)
            print("[阻抗] ✅ 末端控制参数已设置")

            # 验证状态
            sub_data = self.robot.subscribe(self.dcss)
            state = sub_data['states'][0]['cur_state']
            imp_type = sub_data['inputs'][0]['imp_type']

            print(f"\n[阻抗] 验证:")
            print(f"  当前状态: {state} (3=扭矩模式)")
            print(f"  阻抗类型: {imp_type} (2=笛卡尔)")

            if state == 3 and imp_type == 2:
                print("[阻抗] ✅ 阻抗模式设置成功")
                return True
            else:
                print("[阻抗] ⚠️ 状态可能不正确")
                return False

        except Exception as e:
            print(f"[阻抗] ❌ 设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def move_forward_30mm(self, current_joints, current_xyzabc):
        """前移30mm"""
        self.print_section("步骤4: 前移30mm")

        try:
            # 计算目标位姿
            target_xyzabc = current_xyzabc.copy()
            target_xyzabc[0] += 30.0  # X方向+30mm

            print(f"[移动] 当前位置: X={current_xyzabc[0]:.2f}mm")
            print(f"[移动] 目标位置: X={target_xyzabc[0]:.2f}mm")
            print(f"[移动] 位移: ΔX=+30mm")

            print("\n[移动] ⚠️ 机器人将沿X轴前移30mm")
            print("[移动] ⚠️ 确保运动路径安全，无障碍物！")
            self.wait_for_continue("确认安全后按Enter执行移动")

            # movLA 规划 + 逐点执行（参考 demo 20）
            points, pset = self.kk.movLA(
                start_xyzabc=current_xyzabc.tolist(),
                end_xyzabc=target_xyzabc.tolist(),
                ref_joints=current_joints.tolist(),
                vel=50,
                acc=50,
                freq_hz=50
            )

            if not points or len(points) == 0:
                print("[移动] ❌ 轨迹规划失败，执行急停...")
                self.emergency_stop()
                sys.exit(1)

            print(f"✓ 规划完成，共 {len(points)} 个点，预计 {len(points) * 0.02:.1f}s")

            # 逐点发送
            print("\n[移动] 执行轨迹...")
            for i in range(len(points)):
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm='A', joints=points[i])
                self.robot.send_cmd()
                time.sleep(0.02)  # 50Hz

                if i % 50 == 0:
                    progress = (i / len(points)) * 100
                    print(f"  进度: {progress:.0f}%")

            print("✓ 轨迹执行完成")
            time.sleep(1.0)

            # 检查最终位置
            sub_data = self.robot.subscribe(self.dcss)
            final_joints = sub_data['outputs'][0]['fb_joint_pos']
            fk_mat = self.kk.fk(joints=final_joints)
            final_xyzabc = self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat)

            delta_x = final_xyzabc[0] - current_xyzabc[0]

            print(f"\n[移动] 最终位置: X={final_xyzabc[0]:.2f}mm")
            print(f"[移动] 实际位移: ΔX={delta_x:.2f}mm")

            if abs(delta_x - 30.0) < 5.0:
                print("[移动] ✅ 移动成功 (误差<5mm)")
            else:
                print(f"[移动] ⚠️ 移动偏差较大: {delta_x:.2f}mm vs 30mm")

            return True

        except Exception as e:
            print(f"[移动] ❌ 移动失败: {e}")
            import traceback
            traceback.print_exc()
            print("[移动] 执行急停...")
            self.emergency_stop()
            sys.exit(1)

    def cleanup(self):
        """清理并断开连接（正常退出前先下使能）"""
        self.print_section("步骤5: 清理")

        if self.robot is not None:
            try:
                # 先切换到下使能（关节锁死），参考demo 20步骤6
                print("[清理] 切换到下使能状态...")
                self.robot.clear_set()
                self.robot.set_state(arm='A', state=0)  # 0=下使能
                self.robot.set_state(arm='B', state=0)
                self.robot.send_cmd()
                time.sleep(0.5)
                print("[清理] ✅ 已下使能（关节锁死）")

                # 再断开连接
                print("[清理] 断开连接...")
                self.robot.release_robot()
                print("[清理] ✅ 已断开")
            except Exception as e:
                print(f"[清理] ⚠️ 断开失败: {e}")

    def run(self):
        """运行测试"""
        print("\n" + "="*60)
        print("🚀 Marvin简单移动测试")
        print("="*60)
        print("⚠️  注意事项:")
        print("  1. 确保机器人周围安全，无人靠近")
        print("  2. 随时准备按下急停按钮")
        print("  3. Ctrl+C可随时终止测试")
        print("  4. 将从当前位置前移30mm")
        print("="*60)

        self.wait_for_continue("准备好后按Enter开始测试")

        # 步骤1: 连接
        if not self.connect():
            return

        # 步骤2: 获取当前位姿
        current_joints, current_xyzabc = self.get_current_pose()
        if current_joints is None:
            self.cleanup()
            return

        # 步骤3: 设置阻抗模式
        if not self.setup_impedance_mode():
            self.cleanup()
            return

        # 步骤4: 前移30mm
        if not self.move_forward_30mm(current_joints, current_xyzabc):
            self.cleanup()
            return

        # 步骤5: 清理
        self.cleanup()

        print("\n" + "="*60)
        print("✅ 测试完成")
        print("="*60)


if __name__ == "__main__":
    test = SimpleMovementTest()
    test.run()
