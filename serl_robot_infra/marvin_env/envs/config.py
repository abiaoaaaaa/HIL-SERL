"""
Marvin机械臂环境默认配置

参考FrankaEnv的DefaultEnvConfig设计，提供Marvin机器人的默认参数
"""
import numpy as np
from typing import Dict, Callable


class DefaultMarvinEnvConfig:
    """Marvin机械臂环境默认配置类"""

    # ==================== 机器人连接配置 ====================
    ROBOT_IP: str = "192.168.14.190"  # Marvin机器人IP地址
    ARM: str = 'A'  # 使用的机械臂：'A'(左臂) 或 'B'(右臂)

    # ==================== 运动学配置文件 ====================
    # 注意：需要根据实际机型选择正确的配置文件
    # CCS机型: ccs_m6_40.MvKDCfg
    # SRS机型: srs_m6_40.MvKDCfg
    KINE_CONFIG_PATH: str = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

    # ==================== 相机配置 ====================
    # 与FrankaEnv保持一致的相机配置格式
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": {
            "serial_number": "427622274205",
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }

    # 图像裁剪函数字典，键为相机名称，值为裁剪函数
    IMAGE_CROP: Dict[str, Callable] = {
        "wrist_1": lambda img: img[50:-200, 200:-200],
    }

    # ==================== 任务位姿配置 ====================
    # 目标位姿 [X, Y, Z, Roll, Pitch, Yaw]
    # 单位：毫米(位置), 度(姿态)
    # 基于实测初始位姿: X=362.8, Y=408.7, Z=123.7
    TARGET_POSE: np.ndarray = np.array([362.8, 408.7, 123.7, -91.2, 0.8, -84.2])

    # 抓取预备位姿 = 目标位姿上方 50mm
    GRASP_POSE: np.ndarray = np.array([362.8, 408.7, 173.7, -91.2, 0.8, -84.2])

    # 重置位姿（任务开始位置）= 当前位置
    RESET_POSE: np.ndarray = np.array([362.8, 408.7, 123.7, -91.2, 0.8, -84.2])

    # 奖励判定阈值 [X, Y, Z, Roll, Pitch, Yaw]
    # 单位：毫米(位置), 度(姿态)
    REWARD_THRESHOLD: np.ndarray = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

    # ==================== 动作缩放参数 ====================
    # 参考 spacemouse_control: POS_SCALE=5.0mm, ROT_SCALE=0.015rad
    # ACTION_SCALE = [位置缩放(mm), 姿态缩放(rad), 夹爪缩放]
    # 每步最大位移 = 1.0 * ACTION_SCALE[0] = 3mm (默认保守值)
    # 每步最大旋转 = 1.0 * ACTION_SCALE[1] ≈ 0.015rad ≈ 0.86°
    ACTION_SCALE: np.ndarray = np.array([3.0, 0.015, 1.0])

    # ==================== 随机重置配置 ====================
    RANDOM_RESET: bool = False  # 是否启用随机重置
    RANDOM_XY_RANGE: float = 10.0  # XY平面随机范围（毫米）
    RANDOM_RZ_RANGE: float = 0.1  # 绕Z轴随机旋转范围（弧度）

    # ==================== 笛卡尔空间安全边界 ====================
    # 限制末端在笛卡尔空间的运动范围，防止碰撞
    # 以初始位姿为中心 ±100mm × ±100mm × ±100mm 的安全范围
    # [X_min, Y_min, Z_min, Roll_min, Pitch_min, Yaw_min]
    ABS_POSE_LIMIT_LOW: np.ndarray = np.array([262.8, 208.7,  23.7, -101.2, -9.2, -94.2])
    # [X_max, Y_max, Z_max, Roll_max, Pitch_max, Yaw_max]
    ABS_POSE_LIMIT_HIGH: np.ndarray = np.array([562.8, 508.7, 400, -81.2, 10.8, -74.2])

    # ==================== 阻抗控制模式选择 ====================
    # IMPEDANCE_MODE: 控制模式选择
    #   "cartesian" - 笛卡尔阻抗（默认）：末端空间控制，适合精确操作
    #   "joint"     - 关节阻抗：关节空间控制，适合柔顺交互/拖动示教
    IMPEDANCE_MODE: str = "cartesian"

    # ==================== 笛卡尔阻抗参数 ====================
    # 用于末端精确控制（IMPEDANCE_MODE="cartesian"）
    # 刚度K: [Kx, Ky, Kz, Krx, Kry, Krz, Kn]
    # 平移刚度范围: 0-3000 N/m
    # 旋转刚度范围: 0-100 Nm/rad
    # 零空间刚度: 0-20
    COMPLIANCE_PARAM: Dict[str, np.ndarray] = {
        "K": np.array([8000.0, 8000.0, 8000.0, 600.0, 600.0, 600.0, 20.0]),
        "D": np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0]),  # 阻尼比，范围0-1
    }

    # 精密模式参数（reset/抓取移动时使用，高刚度抗下垂）
    PRECISION_PARAM: Dict[str, np.ndarray] = {
        "K": np.array([8000.0, 8000.0, 8000.0, 600.0, 600.0, 600.0, 20.0]),
        "D": np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0]),
    }

    # ==================== 关节阻抗参数 ====================
    # 用于柔顺控制（IMPEDANCE_MODE="joint"）
    # 刚度K: [K1, K2, K3, K4, K5, K6, K7] (N·m/度)
    # 阻尼D: [D1, D2, D3, D4, D5, D6, D7] (N·m/(度/秒))
    # 注意：关节阻抗需更低刚度避免震动，采用低刚度配低阻尼
    JOINT_COMPLIANCE_PARAM: Dict[str, np.ndarray] = {
        "K": np.array([12.0, 12.0, 12.0, 10.0, 9.0, 9.0, 7.0]),  # 关节刚度
        "D": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]),     # 关节阻尼(0-1)
    }

    # ==================== 控制参数 ====================
    DISPLAY_IMAGE: bool = True  # 是否显示图像（调试用）
    GRIPPER_SLEEP: float = 0.2  # 夹爪动作后等待时间（秒）- 优化：从0.6降到0.2以减少延迟
    MAX_EPISODE_LENGTH: int = 100  # 最大episode长度
    JOINT_RESET_PERIOD: int = 0  # 关节重置周期（0表示不启用）

    # 夹爪配置
    USE_GRIPPER: bool = True  # 是否启用夹爪控制
    GRIPPER_MOTOR_ID: int = 1  # 夹爪电机CAN ID

    # 控制频率（Hz）
    # 注意：Marvin支持1KHz通信，但实际控制频率可以设置更低
    HZ: int = 10

    # 速度和加速度百分比（范围0-100）
    VEL_RATIO: int = 10  # 速度百分比，10表示10%
    ACC_RATIO: int = 10  # 加速度百分比，10表示10%

    # ==================== 末端工具配置 ====================
    # 工具运动学参数 [X, Y, Z, Roll, Pitch, Yaw]
    # 相对于末端法兰的偏移（毫米和度）
    TOOL_KINE_PARAMS: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 工具动力学参数（用于重力补偿）
    # [质量(kg), 质心X,Y,Z(mm), 惯量Ixx,Ixy,Ixz,Iyy,Iyz,Izz(kg·mm²)]
    # 参考 demo: 夹爪约 0.5kg, 质心在 Z 方向约 50mm
    TOOL_DYN_PARAMS: np.ndarray = np.array([1, 0.0, 0.0, 50.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.03])
