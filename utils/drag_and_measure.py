#!/usr/bin/env python3
"""
柔顺拖动模式 — 实时读取末端位姿，用于确定安全边界

功能:
1. 连接机器人 → 清错 → 进入关节阻抗模式（低刚度柔顺，可手动拖动）
2. 实时打印当前末端 TCP 位姿 (XYZABC mm/deg)
3. 双触发模式:
   - 键盘模式: Enter → 开始/停止记录，Q → 退出
   - 按钮模式: 按住拖动按钮 → 自动记录，松开 → 停止
4. 使用SDK底层采集，支持高频记录（关节位置、速度、力矩等）
5. Ctrl+C → 急停退出

使用方法:
    python drag_and_measure.py

操作:
    1. 选择触发模式（键盘 or 按钮）
    2. 开始记录后拖动机械臂遍历所有可达位置
    3. 停止记录
    4. 程序自动显示推荐安全边界并保存轨迹数据
"""

import sys
import os
import time
import signal
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../..'))
sys.path.insert(0, os.path.join(current_dir, '../../../serl_robot_infra/marvin_env'))

from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
from SDK_PYTHON.fx_kine import Marvin_Kine

# ==============================================================================
# 配置
# ==============================================================================
ROBOT_IP = "192.168.14.190"
ARM = 'A'
ARM_IDX = 0 if ARM == 'A' else 1
KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

# 关节阻抗参数（低刚度低阻尼 = 柔顺拖动）
JOINT_K = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
JOINT_D = [0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]

# 持续记录时每秒采样次数（键盘模式）
RECORD_HZ = 10
# 界面刷新频率
PRINT_HZ = 5

# 触发模式
TRIGGER_MODE_KEYBOARD = 1  # 键盘Enter键触发
TRIGGER_MODE_BUTTON = 2    # 物理拖动按钮触发

# ==============================================================================
# 全局状态
# ==============================================================================
robot = None
dcss = None
kk = None
running = True
recording = False  # 是否正在持续记录
recorded_poses = []  # 记录的位姿
trigger_mode = TRIGGER_MODE_KEYBOARD  # 默认键盘模式
use_sdk_collect = False  # 是否使用SDK底层高频采集


def emergency_stop(signum=None, frame=None):
    """急停"""
    global running
    print("\n\n" + "=" * 60)
    print("🛑 急停: Ctrl+C")
    print("=" * 60)
    running = False
    if robot is not None:
        try:
            robot.clear_set()
            robot.set_state(arm=ARM, state=0)
            robot.set_state(arm='B', state=0)
            robot.send_cmd()
            time.sleep(0.3)
            print("[急停] ✅ 已下使能")
        except Exception:
            pass
        try:
            robot.release_robot()
            print("[急停] ✅ 已断开连接")
        except Exception:
            pass
    print("=" * 60)
    os._exit(0)


signal.signal(signal.SIGINT, emergency_stop)


def status_label():
    """当前状态标签（函数形式，运行时读取全局变量）"""
    mode_str = "🎹键盘" if trigger_mode == TRIGGER_MODE_KEYBOARD else "🔘按钮"
    if recording:
        return f"🔴 记录中[{mode_str}]"
    else:
        return f"⏸  已暂停[{mode_str}]"


def print_pose(xyzabc):
    """打印当前位姿"""
    sys.stdout.write(
        f"\r  {status_label()} | "
        f"X={xyzabc[0]:7.1f}  Y={xyzabc[1]:7.1f}  Z={xyzabc[2]:7.1f}  "
        f"A={xyzabc[3]:7.1f}  B={xyzabc[4]:7.1f}  C={xyzabc[5]:7.1f}  "
        f"| 已采集: {len(recorded_poses)} 点  |  Enter=录/停  Q=退出"
    )
    sys.stdout.flush()


def print_bounds(poses):
    """打印推荐的安全边界"""
    if len(poses) < 2:
        print("\n⚠  记录点不足 (<2)，无法计算边界")
        return
    arr = np.array(poses)
    low = arr.min(axis=0)
    high = arr.max(axis=0)
    # 加裕度
    margin_pos = 10.0  # mm
    margin_rot = 5.0   # deg
    safe_low = low.copy()
    safe_high = high.copy()
    safe_low[:3] -= margin_pos
    safe_high[:3] += margin_pos
    safe_low[3:] -= margin_rot
    safe_high[3:] += margin_rot

    print("\n" + "=" * 60)
    print("📐 推荐安全边界 (已加裕度)")
    print("=" * 60)
    print(f"  采集点数: {len(poses)}")
    print(f"  实际最小 XYZABC: {np.array2string(low, precision=1)}")
    print(f"  实际最大 XYZABC: {np.array2string(high, precision=1)}")
    print()
    print("  📋 粘贴到 config.py:")
    print(f"  ABS_POSE_LIMIT_LOW = np.{repr(np.round(safe_low, 1))}")
    print(f"  ABS_POSE_LIMIT_HIGH = np.{repr(np.round(safe_high, 1))}")
    print("=" * 60)


def select_arm_dialog():
    """选择要控制的机械臂"""
    print("\n" + "-" * 60)
    print("选择机械臂:")
    print("  A - 左臂")
    print("  B - 右臂")
    choice = input("请输入 A 或 B [A]: ").strip().upper() or 'A'
    if choice == 'B':
        a = 'B'
    else:
        a = 'A'
    print(f"已选择: {a}臂\n")
    return a


def select_trigger_mode_dialog():
    """选择触发模式"""
    print("\n" + "-" * 60)
    print("选择触发模式:")
    print("  1 - 键盘模式 (按Enter键开始/停止记录)")
    print("  2 - 按钮模式 (按住拖动按钮自动记录)")
    choice = input("请输入 1 或 2 [1]: ").strip() or '1'
    if choice == '2':
        mode = TRIGGER_MODE_BUTTON
        print("已选择: 按钮模式")
    else:
        mode = TRIGGER_MODE_KEYBOARD
        print("已选择: 键盘模式")
    print("-" * 60)
    return mode


def select_collection_method_dialog():
    """选择数据采集方法"""
    print("\n" + "-" * 60)
    print("选择数据采集方法:")
    print("  1 - 标准模式 (Python端采集，适合低频率 ~10Hz)")
    print("  2 - 高频模式 (SDK底层采集，支持高频 ~100Hz)")
    choice = input("请输入 1 或 2 [1]: ").strip() or '1'
    if choice == '2':
        use_sdk = True
        print("已选择: 高频模式 (SDK底层采集)")
    else:
        use_sdk = False
        print("已选择: 标准模式 (Python端采集)")
    print("-" * 60)
    return use_sdk


def verify_udp_connection(robot, dcss, timeout=5):
    """验证UDP数据通道连接成功（参考示例代码）"""
    print("  验证UDP数据通道...")
    motion_tag = 0
    frame_update = None
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            sub_data = robot.subscribe(dcss)
            frame_serial = sub_data['outputs'][0]['frame_serial']

            if frame_serial != 0 and frame_update != frame_serial:
                motion_tag += 1
                frame_update = frame_serial

            if motion_tag > 0:
                print(f"  ✅ UDP连接成功 (frame_serial={frame_serial})")
                return True

        except Exception:
            pass

        time.sleep(0.01)

    print("  ✗ UDP连接失败 (可能被防火墙阻止)")
    return False


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    global ARM, ARM_IDX, robot, dcss, kk, running, recording, recorded_poses, trigger_mode, use_sdk_collect

    print("\n" + "=" * 60)
    print("🔧 柔顺拖动模式 — 持续记录末端位姿")
    print("=" * 60)
    print(f"  IP: {ROBOT_IP}")

    try:
        arm_choice = select_arm_dialog()
        ARM = arm_choice
        ARM_IDX = 0 if ARM == 'A' else 1

        trigger_mode = select_trigger_mode_dialog()
        use_sdk_collect = select_collection_method_dialog()
    except KeyboardInterrupt:
        emergency_stop()

    print(f"  关节阻抗 K: {JOINT_K}")
    print(f"  关节阻抗 D: {JOINT_D}")

    if trigger_mode == TRIGGER_MODE_KEYBOARD:
        print(f"  记录频率: {RECORD_HZ} Hz")
        print()
        print("⚠  操作说明:")
        print("  - Enter → 开始/停止记录")
        print("  - 记录中拖动机械臂遍历所有可达位置")
        print("  - Q → 退出并显示推荐安全边界")
    else:
        print()
        print("⚠  操作说明:")
        print("  - 按住拖动按钮 → 自动开始记录")
        print("  - 记录中拖动机械臂遍历所有可达位置")
        print("  - 松开按钮 → 自动停止记录")
        print("  - Q → 退出并显示推荐安全边界")

    print("  - Ctrl+C → 急停退出")
    print()

    # ---- 连接 ----
    print("[1/5] 连接机器人...")
    robot = Marvin_Robot()
    dcss = DCSS()
    ret = robot.connect(ROBOT_IP)
    if ret == 0:
        print("✗ 连接失败 (端口被占用)")
        return
    print("  ✅ 已连接")
    time.sleep(0.5)

    # ---- 验证UDP连接 ----
    if not verify_udp_connection(robot, dcss):
        emergency_stop()

    # ---- 开启日志（便于调试）----
    if trigger_mode == TRIGGER_MODE_BUTTON:
        print("  开启SDK日志...")
        robot.log_switch('1')  # 全局日志
        robot.local_log_switch('1')  # 主要日志

    # ---- 初始化运动学 ----
    print("[2/5] 初始化运动学...")
    kk = Marvin_Kine()
    kk.log_switch(0)
    ini = kk.load_config(arm_type=ARM_IDX, config_path=KINE_CONFIG_PATH)
    if not ini:
        print("✗ 加载配置失败")
        emergency_stop()
    kk.initial_kine(
        robot_type=ini['TYPE'][ARM_IDX],
        dh=ini['DH'][ARM_IDX],
        pnva=ini['PNVA'][ARM_IDX],
        j67=ini['BD'][ARM_IDX]
    )
    print("  ✅ 运动学就绪")

    # ---- 清错 ----
    print("[3/5] 清除错误...")
    robot.check_error_and_clear(dcss)
    time.sleep(0.5)

    # 获取初始位姿
    sub = robot.subscribe(dcss)
    joints = sub['outputs'][ARM_IDX]['fb_joint_pos']
    fk_mat = kk.fk(joints=joints)
    xyzabc = np.array(kk.mat4x4_to_xyzabc(pose_mat=fk_mat))
    print(f"  当前 TCP: X={xyzabc[0]:.1f} Y={xyzabc[1]:.1f} Z={xyzabc[2]:.1f} "
          f"A={xyzabc[3]:.1f} B={xyzabc[4]:.1f} C={xyzabc[5]:.1f}")

    # ---- 进入关节阻抗模式 ----
    print(f"[4/5] 进入关节阻抗模式（柔顺拖动）...")

    # 先下使能确保干净状态
    robot.clear_set()
    robot.set_state(arm=ARM, state=0)
    robot.send_cmd()
    time.sleep(0.5)

    # 设置扭矩模式
    robot.clear_set()
    robot.set_state(arm=ARM, state=3)
    robot.send_cmd()
    time.sleep(1.0)

    # 设置关节阻抗
    robot.clear_set()
    robot.set_impedance_type(arm=ARM, type=1)
    robot.set_joint_kd_params(arm=ARM, K=JOINT_K, D=JOINT_D)
    robot.set_vel_acc(arm=ARM, velRatio=10, AccRatio=10)
    robot.send_cmd()
    time.sleep(0.5)

    # 验证是否进入关节阻抗模式
    sub = robot.subscribe(dcss)
    if sub["states"][ARM_IDX]["cur_state"] != 3:
        print("  ✗ 未进入扭矩模式")
        emergency_stop()
    if sub["inputs"][ARM_IDX]["imp_type"] != 1:
        print("  ✗ 未进入关节阻抗模式")
        emergency_stop()

    print("  ✅ 关节阻抗模式已激活")

    # ---- 设置拖动类型（按钮模式必须）----
    if trigger_mode == TRIGGER_MODE_BUTTON:
        print("[5/5] 设置拖动类型（关节空间拖动）...")
        robot.clear_set()
        robot.set_drag_space(arm=ARM, dgType=1)
        # dgType: 0=退出拖动, 1=关节空间, 2-5=笛卡尔各方向
        robot.send_cmd()
        time.sleep(0.5)

        # 验证拖动类型已设置
        sub = robot.subscribe(dcss)
        if sub["inputs"][ARM_IDX]["drag_sp_type"] != 1:
            print("  ✗ 未设置拖动类型")
            emergency_stop()
        print("  ✅ 拖动类型已设置（关节空间）")
    else:
        print("[5/5] 跳过拖动类型设置（键盘模式不需要）")

    print()
    print("=" * 60)
    if trigger_mode == TRIGGER_MODE_KEYBOARD:
        print("✅ 可拖动机械臂 | Enter=开始记录 | Q=退出")
    else:
        print("✅ 可拖动机械臂 | 按住拖动按钮=自动记录 | Q=退出")
        print("   等待按钮按下...")
    print("=" * 60)

    # ---- 主循环 ----
    import select as _select
    import termios
    import tty

    old_settings = termios.tcgetattr(sys.stdin)
    last_record_time = 0
    sdk_collection_started = False
    button_was_pressed = False
    stage_waiting_button = (trigger_mode == TRIGGER_MODE_BUTTON)  # 按钮模式：等待首次按下

    try:
        tty.setcbreak(sys.stdin.fileno())

        while running:
            loop_start = time.time()

            # 读取状态
            sub = robot.subscribe(dcss)
            joints = sub['outputs'][ARM_IDX]['fb_joint_pos']
            fk_mat = kk.fk(joints=joints)
            xyzabc = np.array(kk.mat4x4_to_xyzabc(pose_mat=fk_mat))

            # 按钮模式：检测拖动按钮状态
            if trigger_mode == TRIGGER_MODE_BUTTON:
                button_pressed = sub['outputs'][ARM_IDX]['tip_di'][0] == 1

                # 等待首次按下（参考示例的stage1逻辑）
                if stage_waiting_button:
                    # 验证完整状态链：扭矩模式 → 关节阻抗 → 拖动类型 → 按钮
                    if (sub["states"][ARM_IDX]["cur_state"] == 3 and
                        sub["inputs"][ARM_IDX]["imp_type"] == 1 and
                        sub["inputs"][ARM_IDX]["drag_sp_type"] == 1 and
                        button_pressed):

                        # 状态链验证成功，开始记录
                        recording = True
                        button_was_pressed = True
                        stage_waiting_button = False
                        last_record_time = time.time()
                        recorded_poses.append(xyzabc.tolist())

                        if use_sdk_collect:
                            # 按下按钮后才配置SDK采集（参考示例逻辑）
                            cols = 7
                            idx = [0, 1, 2, 3, 4, 5, 6,  # 关节位置
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0]
                            rows = 1000000
                            robot.clear_set()
                            robot.collect_data(targetNum=cols, targetID=idx, recordNum=rows)
                            robot.send_cmd()
                            time.sleep(0.5)
                            sdk_collection_started = True
                            print("\n  🔴 检测到拖动按钮，开始SDK高频采集...")
                        else:
                            print("\n  🔴 检测到拖动按钮，开始记录...")

                # 已在记录中：检测松开
                elif button_was_pressed and not button_pressed:
                    recording = False
                    button_was_pressed = False

                    if use_sdk_collect and sdk_collection_started:
                        # 停止SDK采集
                        robot.clear_set()
                        robot.stop_collect_data()
                        robot.send_cmd()
                        time.sleep(0.5)
                        sdk_collection_started = False

                    print(f"\n  ⏸  松开按钮，停止采集，已采集 {len(recorded_poses)} 点")
                    print("     按Q退出，或再次按按钮继续记录")

                # 继续按住：持续更新状态
                elif button_pressed:
                    button_was_pressed = True

            # 持续记录（键盘模式 or 按钮按下时）
            if recording and (loop_start - last_record_time) >= (1.0 / RECORD_HZ):
                recorded_poses.append(xyzabc.tolist())
                last_record_time = loop_start

            # 打印（仅在非等待按钮状态时打印，或显示等待状态）
            if stage_waiting_button:
                # 等待按钮时显示状态检查
                sys.stdout.write(
                    f"\r  等待按钮 | 扭矩={sub['states'][ARM_IDX]['cur_state']==3} "
                    f"阻抗={sub['inputs'][ARM_IDX]['imp_type']==1} "
                    f"拖动={sub['inputs'][ARM_IDX]['drag_sp_type']==1} "
                    f"按钮={sub['outputs'][ARM_IDX]['tip_di'][0]==1}  "
                )
                sys.stdout.flush()
            else:
                print_pose(xyzabc)

            # 检查按键
            if _select.select([sys.stdin], [], [], 0.01)[0]:
                key = sys.stdin.read(1)

                if key in ('\r', '\n') and trigger_mode == TRIGGER_MODE_KEYBOARD:   # Enter - 仅键盘模式
                    recording = not recording
                    if recording:
                        last_record_time = time.time()
                        recorded_poses.append(xyzabc.tolist())  # 立即采第一点

                        if use_sdk_collect and not sdk_collection_started:
                            cols = 7
                            idx = [0, 1, 2, 3, 4, 5, 6,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0]
                            rows = 1000000
                            robot.clear_set()
                            robot.collect_data(targetNum=cols, targetID=idx, recordNum=rows)
                            robot.send_cmd()
                            time.sleep(0.5)
                            sdk_collection_started = True

                        print(f"\n  🔴 开始持续记录 ({RECORD_HZ} Hz)...")
                    else:
                        if use_sdk_collect and sdk_collection_started:
                            # 停止SDK采集
                            robot.clear_set()
                            robot.stop_collect_data()
                            robot.send_cmd()
                            time.sleep(0.5)
                            sdk_collection_started = False

                        print(f"\n  ⏸  停止记录，已采集 {len(recorded_poses)} 点")

                elif key.lower() == 'q':
                    print("\n\n退出采集...")
                    if use_sdk_collect and sdk_collection_started:
                        robot.clear_set()
                        robot.stop_collect_data()
                        robot.send_cmd()
                        time.sleep(0.5)
                    break

            # 控制循环频率
            if stage_waiting_button:
                # 等待按钮时快速检查
                time.sleep(0.01)
            else:
                # 正常运行时控制打印频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0 / PRINT_HZ - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    # ---- 打印结果 ----
    print_bounds(recorded_poses)

    # ---- 保存采集数据 ----
    if use_sdk_collect and len(recorded_poses) > 0:
        print("\n[保存] 保存SDK采集数据...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f'drag_trajectory_{timestamp}.txt'
        try:
            robot.save_collected_data_to_path(save_path)
            print(f"  ✅ SDK数据已保存: {save_path}")
            time.sleep(2)  # 等待保存完成
        except Exception as e:
            print(f"  ⚠  SDK数据保存失败: {e}")

    # 保存位姿数据
    if len(recorded_poses) > 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pose_path = f'drag_poses_{timestamp}.txt'
        try:
            np.savetxt(pose_path, recorded_poses, fmt='%.3f',
                      header='X(mm) Y(mm) Z(mm) A(deg) B(deg) C(deg)',
                      comments='# ')
            print(f"  ✅ 位姿数据已保存: {pose_path}")
        except Exception as e:
            print(f"  ⚠  位姿数据保存失败: {e}")

    # ---- 下使能 + 断开 ----
    print("\n[清理] 下使能...")
    try:
        # 按钮模式：退出拖动
        if trigger_mode == TRIGGER_MODE_BUTTON:
            robot.clear_set()
            robot.set_drag_space(arm=ARM, dgType=0)
            robot.send_cmd()
            time.sleep(0.5)
            print("  ✅ 已退出拖动模式")

        robot.clear_set()
        robot.set_state(arm=ARM, state=0)
        robot.set_state(arm='B', state=0)
        robot.send_cmd()
        time.sleep(0.3)
        robot.release_robot()
        print("[清理] ✅ 已断开")
    except Exception:
        pass


if __name__ == "__main__":
    main()
