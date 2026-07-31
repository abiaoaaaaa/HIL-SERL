#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估动作轨迹回放器 — 加载 eval_actions_*.json 在真实机器人上回放

用法:
    cd /home/xlb/code_marvin/hil-serl

    # 回放指定episode
    python utils/debug_tools/debug_movla_spacemouse.py --episode 0

    # 回放全部episode
    python utils/debug_tools/debug_movla_spacemouse.py --episode -1

    # 指定JSON文件
    python utils/debug_tools/debug_movla_spacemouse.py --json <path> --episode 0

    按 Ctrl+C 提前中断当前episode
"""

import sys
import os
import time
import signal
import numpy as np
import json
import argparse

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
# 默认配置（与 env config.py 保持一致）
# ==============================================================================

ROBOT_IP = "192.168.14.190"
ARM = 'A'
KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

DEFAULT_JSON_PATH = (
    "/home/xlb/code_marvin/hil-serl/examples/experiments/"
    "marvin_usb_insertion/checkpoints/eval_actions_124000.json"
)

# 动作缩放参数（与 config.ACTION_SCALE 一致）
POS_SCALE = 20.0       # mm per unit (action_scale[0])
ROT_SCALE = 0.05       # rad per unit (action_scale[1])

MOVLA_VEL = 100
MOVLA_ACC = 100
MOVLA_FREQ_HZ = 100

# 安全范围
SAFETY_LOW = np.array([342.2, 258.1, 109.1, -109.1, -16.3, -103.2])
SAFETY_HIGH = np.array([516.5, 390.1, 381.9, -77.5, 23.5, -71.3])

# 默认 RESET_POSE（从 config.py）
RESET_POSE = np.array([394.3, 321.7, 200.3, -90.0005, 0.001, -90.001])

GRIPPER_MOTOR_ID = 1
GRIPPER_SLEEP = 0.2


# ==============================================================================
# 轨迹回放器
# ==============================================================================

class TrajectoryReplayer:
    """加载 eval_actions JSON 并在机器人上回放，执行逻辑对标 MarvinEnv._execute_sub_step"""

    def __init__(self, json_path: str, episode: int):
        self.json_path = json_path
        self.target_episode = episode  # -1 = 全部

        self.robot = None
        self.kk = None
        self.dcss = None
        self.gripper = None
        self.arm = ARM
        self.arm_idx = 0 if ARM == 'A' else 1
        self.running = False

    def initialize(self):
        """初始化机器人（与 env 一致）"""
        print(f"\n[初始化] 连接机器人 {ROBOT_IP}...")
        self.robot = Marvin_Robot()
        self.dcss = DCSS()
        ret = self.robot.connect(ROBOT_IP)
        if ret == 0:
            raise RuntimeError(f"Failed to connect to robot at {ROBOT_IP}")
        print("[初始化] ✓ 连接成功")
        time.sleep(0.5)

        # 运动学
        print("[初始化] 加载运动学配置...")
        self.kk = Marvin_Kine()
        self.kk.log_switch(0)
        ini_result = self.kk.load_config(arm_type=self.arm_idx, config_path=KINE_CONFIG_PATH)
        if not ini_result:
            raise RuntimeError("Failed to load kinematics config")
        initial_tag = self.kk.initial_kine(
            robot_type=ini_result['TYPE'][self.arm_idx],
            dh=ini_result['DH'][self.arm_idx],
            pnva=ini_result['PNVA'][self.arm_idx],
            j67=ini_result['BD'][self.arm_idx],
        )
        if not initial_tag:
            raise RuntimeError("Failed to initialize kinematics")
        print("[初始化] ✓ 运动学初始化成功")

        # 夹爪
        print("[初始化] 初始化夹爪...")
        try:
            from marvin_env.gripper.marvin_gripper import MarvinGripperController
            self.gripper = MarvinGripperController(
                robot=self.robot, arm=self.arm, motor_id=GRIPPER_MOTOR_ID
            )
            if self.gripper.initialize():
                print("[初始化] ✓ 夹爪初始化成功")
            else:
                print("[初始化] ✗ 夹爪初始化失败")
                self.gripper = None
        except Exception as e:
            print(f"[初始化] ✗ 夹爪初始化失败: {e}")
            self.gripper = None

        # 切换到柔顺模式（对标 env._enter_compliance_mode）
        print("[初始化] 切换到笛卡尔柔顺模式...")
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

    def _update_currpos(self):
        """更新当前关节和笛卡尔位姿（对标 env._update_currpos）"""
        sub = self.robot.subscribe(self.dcss)
        self.curr_joints = np.array(sub['outputs'][self.arm_idx]['fb_joint_pos'])
        fk_mat = self.kk.fk(joints=self.curr_joints.tolist())
        self.currpos_xyzabc = np.array(self.kk.mat4x4_to_xyzabc(pose_mat=fk_mat))

    def _send_gripper_command(self, gripper_action: float):
        """发送夹爪指令（对标 env._send_gripper_command）"""
        if self.gripper is None:
            return
        try:
            # gripper_action > 0 → open, < 0 → close
            if gripper_action > 0.5:
                self.gripper.open()
            elif gripper_action < -0.5:
                self.gripper.close()
        except Exception as e:
            print(f"[夹爪] 指令失败: {e}")

    def execute_action(self, action: np.ndarray):
        """
        执行单步动作，完全对标 MarvinEnv._execute_sub_step
        action: [dx, dy, dz, drz, gripper], 范围 [-1, 1]
        """
        # 1. 更新当前位置
        self._update_currpos()
        current_xyzabc = self.currpos_xyzabc.copy()

        # 2. 计算目标位姿（对标 env._execute_sub_step:1047-1058）
        pos_delta_mm = action[:3] * POS_SCALE  # mm
        target_xyzabc = current_xyzabc.copy()
        target_xyzabc[:3] = current_xyzabc[:3] + pos_delta_mm

        # 姿态增量（只处理 drz）
        action_rot_rad = np.zeros(3)
        action_rot_rad[2] = action[3] * ROT_SCALE
        target_xyzabc[3:] = current_xyzabc[3:] + np.rad2deg(action_rot_rad)

        # 强制锁定 A/B 旋转到 RESET_POSE
        target_xyzabc[3] = RESET_POSE[3]
        target_xyzabc[4] = RESET_POSE[4]

        # 安全裁剪
        target_xyzabc = self._clip_safety_box(target_xyzabc)

        # 3. movLA 规划（对标 env:1131-1137）
        delta_dist_mm = np.linalg.norm(pos_delta_mm)
        delta_rot_rad = np.linalg.norm(action_rot_rad)

        if delta_dist_mm > 0.01 or delta_rot_rad > 1e-5:
            points, _ = self.kk.movLA(
                start_xyzabc=current_xyzabc.tolist(),
                end_xyzabc=target_xyzabc.tolist(),
                ref_joints=self.curr_joints.tolist(),
                vel=MOVLA_VEL,
                acc=MOVLA_ACC,
                freq_hz=MOVLA_FREQ_HZ,
            )
        else:
            points = None

        # 4. 发送轨迹点（对标 env:1148-1151）
        if points and len(points) > 0:
            for pt in points:
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                self.robot.send_cmd()

        # 5. 夹爪指令
        self._send_gripper_command(action[4])

        # 6. 自适应 sleep（对标 env:1194-1204）
        # 简单版：固定 50ms
        time.sleep(0.05)

        return {
            "current_xyzabc": current_xyzabc.tolist(),
            "target_xyzabc": target_xyzabc.tolist(),
            "pos_delta_mm": pos_delta_mm.tolist(),
            "num_planned_points": len(points) if points else 0,
        }

    def _clip_safety_box(self, target_xyzabc: np.ndarray) -> np.ndarray:
        """安全裁剪"""
        clipped = target_xyzabc.copy()
        clipped[:3] = np.clip(clipped[:3], SAFETY_LOW[:3], SAFETY_HIGH[:3])
        return clipped

    def go_to_reset(self):
        """回到 RESET_POSE"""
        print("[go_to_reset] 打开夹爪...")
        self._send_gripper_command(1.0)
        time.sleep(GRIPPER_SLEEP)

        # 停止运动
        self.robot.stopRunPln_joint(arm=self.arm)
        time.sleep(0.5)

        # 切位置模式
        for _ in range(5):
            self.robot.clear_set()
            self.robot.set_state(arm=self.arm, state=1)
            self.robot.set_vel_acc(arm=self.arm, velRatio=50, AccRatio=50)
            self.robot.send_cmd()
            time.sleep(0.3)
            sub = self.robot.subscribe(self.dcss)
            if sub['states'][self.arm_idx]['cur_state'] == 1:
                print("[go_to_reset] ✓ 切换到位置模式")
                break
            time.sleep(0.5)

        # 移动到 RESET_POSE
        print(f"[go_to_reset] 移动到 RESET_POSE: {RESET_POSE[:3]}")
        self._update_currpos()
        current = self.currpos_xyzabc.copy()
        points, _ = self.kk.movLA(
            start_xyzabc=current.tolist(),
            end_xyzabc=RESET_POSE.tolist(),
            ref_joints=self.curr_joints.tolist(),
            vel=100, acc=100, freq_hz=100,
        )
        if points:
            for pt in points:
                self.robot.clear_set()
                self.robot.set_joint_cmd_pose(arm=self.arm, joints=pt)
                self.robot.send_cmd()
                time.sleep(0.002)
        time.sleep(0.5)

    def run(self):
        """主回放循环"""
        # 加载 JSON
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        episodes = data['episodes']

        # 筛选 episode
        if self.target_episode >= 0:
            matching = [ep for ep in episodes if ep['episode'] == self.target_episode]
            if not matching:
                print(f"[错误] 未找到 episode={self.target_episode}")
                return
            episodes = matching

        print(f"\n{'='*70}")
        print(f"回放配置:")
        print(f"  JSON: {self.json_path}")
        print(f"  Checkpoint: {data['checkpoint_step']}")
        print(f"  成功率: {data['success_rate']:.1%}")
        print(f"  回放 episode: {[ep['episode'] for ep in episodes]}")
        print(f"  动作缩放: pos={POS_SCALE}mm, rot={ROT_SCALE}rad")
        print(f"  按 Ctrl+C 提前中断当前 episode")
        print(f"{'='*70}")

        for ep in episodes:
            actions = ep['actions']
            ep_id = ep['episode']
            success_label = "✓" if ep['success'] else "✗"

            print(f"\n{'='*70}")
            print(f"[Ep{ep_id}] 开始回放 ({ep['steps']:d} steps, 成功={success_label})")
            print(f"{'='*70}")

            self.running = True
            step_count = 0
            start_time = time.time()

            for a in actions:
                if not self.running:
                    break

                action = np.array(a, dtype=np.float64)
                print(f"[Ep{ep_id} step={step_count:3d}] "
                      f"dx={action[0]:+.4f} dy={action[1]:+.4f} dz={action[2]:+.4f} "
                      f"drz={action[3]:+.4f} gripper={action[4]:+.4f}")

                self.execute_action(action)
                step_count += 1

            elapsed = time.time() - start_time
            print(f"[Ep{ep_id}] 回放完成: {step_count} steps, 耗时={elapsed:.1f}s")
            self.running = False

    def shutdown(self):
        print("\n[关闭] 正在关闭...")
        self.running = False
        if self.gripper:
            try:
                self.gripper.shutdown()
            except:
                pass
        if self.robot:
            try:
                self.robot.release_robot()
            except:
                pass
        print("[关闭] 完成")


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='评估动作轨迹回放器')
    parser.add_argument('--json', type=str, default=DEFAULT_JSON_PATH,
                        help=f'动作 JSON 文件路径 (默认: {DEFAULT_JSON_PATH})')
    parser.add_argument('--episode', type=int, default=0,
                        help='回放的 episode 编号 (-1=全部, 默认: 0)')
    args = parser.parse_args()

    replayer = TrajectoryReplayer(json_path=args.json, episode=args.episode)

    def signal_handler(sig, frame):
        print("\n[信号] Ctrl+C，停止回放...")
        replayer.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        replayer.initialize()
        replayer.go_to_reset()
        replayer.run()
    except KeyboardInterrupt:
        print("\n[中断] 用户中断")
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
    finally:
        replayer.shutdown()


if __name__ == "__main__":
    main()
