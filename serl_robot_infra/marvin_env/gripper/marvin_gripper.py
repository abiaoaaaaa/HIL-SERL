"""
Marvin机械臂夹爪控制模块

基于达妙DM4310电机的夹爪控制封装
通过Marvin SDK的CAN通信控制夹爪

MIT模式: q 是弧度值, 范围 ±12.5 rad (DM4310)
参考: DM4310电机位置值和物理位置需要根据实际标定

实测物理范围: 0.0 rad (打开) ~ 1.2 rad (闭合)
"""
import time
import numpy as np
from typing import Optional


class MarvinGripperController:
    """
    Marvin夹爪控制器

    使用达妙DM4310电机，通过CAN总线控制
    MIT模式: 位置 q 为弧度值, 范围约 ±12.5 rad
    """

    def __init__(self, robot, arm: str = 'A', motor_id: int = 1, motor_type=None):
        self.robot = robot
        self.arm = arm
        self.motor_id = motor_id

        try:
            from marvin_env.gripper.KM_CAN import KMGripperControl, Motor, KM_Motor_Type, Control_Type
            self.Control_Type = Control_Type
            self.KMGripperControl = KMGripperControl
            self.Motor = Motor
            self.KM_Motor_Type = KM_Motor_Type
        except ImportError as e:
            raise ImportError(f"无法导入KM_CAN模块: {e}")

        self.gripper_control = None
        self.motor = None

        if motor_type is None:
            self.motor_type = self.KM_Motor_Type.DM4310
        else:
            self.motor_type = motor_type

        # 夹爪位置 (弧度值, 实测标定结果)
        # 电机物理范围: 0.3 ~ 1.2 rad
        # 0.3 rad = 打开, 1.2 rad = 闭合
        self.OPEN_RAD = 0.3       # 打开位置 (rad)
        self.CLOSE_RAD = 1.2      # 闭合位置 (rad, 物理极限)

        # MIT控制参数 - 优化为更高刚度，确保夹爪快速响应
        self.KP = 4.0   # 刚度（增加以提高响应速度）
        self.KD = 0.4    # 阻尼（增加以减少震荡）
        self.CLOSE_KP = 8.0   # 夹紧刚度（更高，确保夹紧力）
        self.CLOSE_KD = 0.2    # 夹紧阻尼

        self.current_position = 0.0
        self.is_enabled = False

        print(f"[MarvinGripper] 夹爪控制器已创建 (ARM={arm}, ID={motor_id})")

    def initialize(self) -> bool:
        """
        初始化夹爪（创建控制对象并使能电机）

        参考 MarvinRobotWrapper._init_grippers:
        disable -> switchControlMode(MIT) -> enable
        """
        try:
            # 创建夹爪控制对象
            self.gripper_control = self.KMGripperControl(robot=self.robot)

            # 创建电机对象
            master_id = self.motor_id + 0x10
            self.motor = self.Motor(self.motor_type, self.motor_id, master_id)

            # 添加电机到控制器
            self.gripper_control.addMotor(self.motor)

            # 选择通道（A臂用'left', B臂用'right'）
            channel = 'left' if self.arm == 'A' else 'right'
            self.gripper_control.add_to_ch(self.motor, channel)

            print(f"[MarvinGripper] 夹爪控制对象已创建")

            # 先禁用 -> 切换MIT模式 -> 再使能 (参考 MarvinRobotWrapper)
            self.gripper_control.disable(self.motor)
            time.sleep(0.3)

            self.gripper_control.switchControlMode(self.motor, self.Control_Type.MIT)
            time.sleep(0.1)

            self.gripper_control.enable(self.motor)
            time.sleep(0.3)
            self.is_enabled = True

            print(f"[MarvinGripper] 电机已使能 (MIT模式)")

            # 读取初始位置
            self._update_position()

            return True

        except Exception as e:
            print(f"[MarvinGripper] 初始化失败: {e}")
            return False

    def open(self, blocking: bool = True) -> bool:
        """打开夹爪"""
        print(f"[MarvinGripper] 🟢 open() 被调用 (blocking={blocking})")
        return self._move_to_position(self.OPEN_RAD, blocking)

    def close(self, blocking: bool = True) -> bool:
        """关闭夹爪（高刚度夹紧）"""
        print(f"[MarvinGripper] 🔴 close() 被调用 (blocking={blocking})")
        return self._move_to_position(self.CLOSE_RAD, blocking, use_higher_stiffness=True)

    def _move_to_position(self, target_rad: float, blocking: bool = False,
                          use_higher_stiffness: bool = False) -> bool:
        """移动到指定位置 (弧度)"""
        if not self.is_enabled or self.gripper_control is None:
            print("[MarvinGripper] 警告: 夹爪未初始化")
            return False

        # 🔍 安全检查：限制目标位置在物理范围内
        target_rad = np.clip(target_rad, self.OPEN_RAD, self.CLOSE_RAD)

        try:
            kp = self.CLOSE_KP if use_higher_stiffness else self.KP
            kd = self.CLOSE_KD if use_higher_stiffness else self.KD

            # 优化：增加非阻塞模式的命令数，从 5 增加到 15
            # 15 × 50ms = 750ms，给电机更多时间到达目标位置
            num_commands = 20 if blocking else 15

            print(f"[MarvinGripper] 开始移动: 目标={target_rad:.2f}rad, "
                  f"当前={self.current_position:.3f}rad, "
                  f"命令数={num_commands}, kp={kp}, kd={kd}")

            start_time = time.time()

            for i in range(num_commands):
                self.gripper_control.controlMIT(
                    self.motor, kp=kp, kd=kd,
                    q=target_rad, dq=0, tau=0
                )
                time.sleep(0.05)

                self.gripper_control.recv()
                self.current_position = self.motor.getPosition()

                # 🔍 DEBUG: 打印关键步骤的位置（首次、中间、最后）
                if i == 0 or i == num_commands // 2 or i == num_commands - 1:
                    print(f"    [step {i:2d}/{num_commands}] 位置={self.current_position:.3f}rad")

            elapsed = time.time() - start_time
            distance = abs(self.current_position - target_rad)
            print(f"[MarvinGripper] 移动完成: 目标={target_rad:.2f}rad, "
                  f"实际={self.current_position:.3f}rad, "
                  f"误差={distance:.3f}rad, 耗时={elapsed*1000:.1f}ms")

            # ⚠️ 警告：如果误差过大
            if distance > 0.15:
                print(f"[MarvinGripper] ⚠️ 警告: 位置误差较大 ({distance:.3f}rad > 0.15rad)")

            return True

        except Exception as e:
            print(f"[MarvinGripper] 移动失败: {e}")
            return False

    def get_position(self) -> float:
        """获取当前夹爪位置 (弧度)"""
        self._update_position()
        return self.current_position

    def get_normalized_position(self) -> float:
        """归一化位置 [0, 1], 0=闭合, 1=打开
        基于实测值简单映射: 0 -> 0.0, OPEN_RAD -> 1.0"""
        # 🔧 先从电机读取最新位置，再计算归一化值
        self._update_position()
        normalized = max(0.0, min(1.0,
            (self.current_position - self.CLOSE_RAD) / (self.OPEN_RAD - self.CLOSE_RAD)))
        # 🔍 DEBUG: 仅在调试时打印（频繁调用，所以注释掉）
        # print(f"[MarvinGripper] get_normalized_position: 原始={self.current_position:.3f}rad -> 归一化={normalized:.3f}")
        return normalized

    def _update_position(self):
        """更新当前位置（从电机读取）"""
        if self.gripper_control is not None and self.motor is not None:
            try:
                self.gripper_control.recv()
                self.current_position = self.motor.getPosition()
            except:
                pass  # 静默失败，保持上一次的值

    def shutdown(self) -> bool:
        """
        关闭夹爪控制（失能电机）

        Returns:
            是否成功关闭
        """
        if not self.is_enabled or self.gripper_control is None:
            return True

        try:
            self.gripper_control.disable(self.motor)
            self.is_enabled = False
            print(f"[MarvinGripper] 电机已失能")
            return True
        except Exception as e:
            print(f"[MarvinGripper] 关闭失败: {e}")
            return False
