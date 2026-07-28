"""
Marvin USB Pickup and Insertion 任务配置

基于Marvin机械臂的USB插拔任务配置
参考: franka的usb_pickup_insertion任务
"""
import os
import sys
import jax
import numpy as np
import jax.numpy as jnp

# 添加路径以支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
serl_infra_path = os.path.join(project_root, 'serl_robot_infra')
examples_path = os.path.join(project_root, 'examples')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if serl_infra_path not in sys.path:
    sys.path.insert(0, serl_infra_path)
if examples_path not in sys.path:
    sys.path.insert(0, examples_path)

from marvin_env.envs.config import DefaultMarvinEnvConfig
from experiments.config import DefaultTrainingConfig


class MarvinUSBEnvConfig(DefaultMarvinEnvConfig):
    """Marvin USB任务环境配置"""

    # ==================== 机器人配置 ====================
    ROBOT_IP = "192.168.14.190"
    ARM = 'A'  # 使用A臂

    # ==================== 运动学配置 ====================
    # TODO: 修改为你的实际路径
    KINE_CONFIG_PATH = "/home/xlb/code_marvin/hil-serl/serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg"

    # ==================== 相机配置 ====================
    # TODO: 修改为你的实际相机序列号
    # 注意：side_policy和side_classifier共享同一个相机，side_policy必须在前面以支持相机共享
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "427622274205",
            "dim": (640, 480),  # RealSense标准分辨率
            "fps": 30,
            "exposure": 8000,
        },
        "wrist_2": {
            "serial_number": "427622272953",
            "dim": (640, 480),  # RealSense标准分辨率
            "fps": 30,
            "exposure": 8000,
        },
        # side_policy必须在side_classifier之前（共享相机）
        "side_policy": {
            "serial_number": "036422060870",
            "dim": (1280, 720),  # RealSense标准分辨率
            "fps": 30,
            "exposure": 8000,
        },
        "side_classifier": {
            "serial_number": "036422060870",  # 共享side_policy的相机
            "dim": (1280, 720),  # RealSense标准分辨率
            "fps": 30,
            "exposure": 8000,
        },
    }

    # 图像裁剪配置
    IMAGE_CROP = {
        "wrist_1": lambda img: img[185:-41, 183:-142],
        "wrist_2": lambda img: img[92:-60, 231:-60],
        "side_policy": lambda img: img[242:-105, 495:-110],
        "side_classifier": lambda img: img[362:-251, 493:-445],
    }

    # ==================== 任务位姿配置 ====================
    # 注意：单位为毫米(mm)和度(°)

    # # USB插座目标位置 (任务目标)
    # TARGET_POSE = np.array([500.0, 200.0, 280.0, 180.0, 0.0, -90.0])

    # # USB抓取预备位置 (在USB上方)
    # GRASP_POSE = np.array([450.0, 150.0, 320.0, 180.0, 0.0, -90.0])

    # 重置位置 (episode开始位置)
    # RESET_POSE = np.array([356.6, 341.9, 295.0, -91.5, 1.5, -89.4])
    # RESET_POSE = np.array([440.3, 314.6, 302.8, 87.1, 0.0, -94.0])
    RESET_POSE = np.array([ 394.3,  321.7,  200.3,  -90.0005,    0.001,  -90.001])

    # 任务完成判定阈值
    REWARD_THRESHOLD = np.array([8.0, 8.0, 8.0, 5.0, 5.0, 5.0])

    # ==================== 末端姿态锁定 ====================
    # 是否锁定末端姿态（True=始终使用 RESET_POSE 的姿态，忽略策略旋转输出）
    FIXED_ORIENTATION = False
    # 锁定的姿态值 [A, B, C] (度)，None 则使用 RESET_POSE[3:]
    FIXED_ORIENTATION_ABC = None

    # ==================== 动作缩放 ====================
    # 参考 spacemouse_control: POS_SCALE=5.0mm, ROT_SCALE=0.015rad
    # [位置缩放(mm), 姿态缩放(rad), 夹爪缩放]
    # 每步最大位移 = 1.0 * ACTION_SCALE[0] ≈ 10mm
    # 每步最大旋转 = 1.0 * ACTION_SCALE[1] ≈ 0.05rad ≈ 2.86°
    ACTION_SCALE = np.array([20.0, 0.05, 1.0])

    # ==================== State 归一化参数 ====================
    # 用于将不同量级的状态量归一化到相似范围
    # 格式: {"mean": [...], "std": [...]}
    # 归一化公式: normalized = (raw - mean) / std
    STATE_NORMALIZATION = {
        "tcp_pose": {
            "mean": [0.4, 0.3, 0.25, -1.5, 0.0, -1.5],  # [x,y,z,rx,ry,rz] euler (Quat2Euler之后)
            "std": [0.15, 0.15, 0.15, 0.5, 0.1, 0.5],
        },
        "tcp_vel": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # [vx,vy,vz,wx,wy,wz]
            "std": [0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
        },
        "tcp_force": {
            "mean": [0.0, 0.0, 0.0],  # [Fx,Fy,Fz]
            "std": [10.0, 10.0, 10.0],
        },
        "tcp_torque": {
            "mean": [0.0, 0.0, 0.0],  # [Tx,Ty,Tz]
            "std": [2.5, 2.5, 2.5],
        },
        "gripper_pose": {
            "mean": [0.0],
            "std": [1.0],  # 已经在[-1,1]范围内
        },
        "last_action": {
            "mean": [0.0, 0.0, 0.0, 0.0, 0.0],  # [dx,dy,dz,dry,gripper]
            "std": [1.0, 1.0, 1.0, 1.0, 1.0],   # 已经在[-1,1]范围内
        },
    }

    # ==================== 随机重置 ====================
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 8.0  # XY平面随机范围(mm)
    RANDOM_RZ_RANGE = 0.08  # 绕Z轴随机范围(弧度)
# ####10hz 250HZ[_get_obs] get_im=1.0ms | subscribe=0.2ms | deepcopy=0.3ms
# [step=146][TIMING] total=100.6ms | update1=0.1 recover=0.0 movLA=2.1 send=0.2 mode=0.0 grip=0.0 sleep=96.1 update2=0.3 obs=1.6 reward=0.2 final=0.0
# Plan MOVLA successful, got 232 points

###20hz  100HZ： [step=193][TIMING] total=66.8ms | update1=0.2 recover=0.0 movLA=0.3 send=0.1 mode=0.0 grip=0.0 sleep=48.0 update2=0.2 obs=18.0 reward=0.1 final=0.0
# Plan MOVLA successful, got 72 points
# [GRIPPER_DEBUG] 输入动作=0.00, 归一化位置=0.13, 原

    # ==================== 安全边界 ====================
    # ABS_POSE_LIMIT_LOW = np.array([ 254.8,  150.7,  153.5, -154.3,  -58.5, -144.2])
    # ABS_POSE_LIMIT_HIGH = np.array([751.3, 551.4, 504.1, -63.2,  51.3, -72.1])
    # ABS_POSE_LIMIT_LOW = np.array([ 342.2,  258.1,  109.1, -109.1,  -16.3, -103.2])
    # ABS_POSE_LIMIT_HIGH = np.array([516.5, 390.1, 381.9, -77.5,  23.5, -71.3])
    ABS_POSE_LIMIT_LOW = np.array([ 325.8,  274.1,  123.1, -111.9,  -11.9,  -99. ])
    ABS_POSE_LIMIT_HIGH = np.array([518.6, 400.3, 405.7, -79.2,  12.8, -76.6])
    # ==================== 阻抗控制模式 ====================
    # IMPEDANCE_MODE: Step阶段控制模式选择
    #   "cartesian" - 笛卡尔阻抗（默认）：末端空间控制，适合精确插入任务
    #   "joint"     - 关节阻抗：关节空间控制，适合柔顺交互，低刚度可推动
    IMPEDANCE_MODE = "cartesian"  # 切换为 "joint" 启用关节阻抗

    # ==================== 笛卡尔阻抗参数 ====================
    # 用于IMPEDANCE_MODE="cartesian"时的step控制
    # 参考 demo07 (柔顺): K=[10,5000,5000,600,600,600,20], D=[0.1*3,0.3*3,1]
    # 参考 demo20 (移动): K=[8000,8000,8000,600,600,600,20], D=[0.8*3,0.4*3,1]

    # 柔顺模式（RL训练时使用，X方向柔顺可推动）
    COMPLIANCE_PARAM = {
        "K": np.array([6000.0, 6000.0, 6000.0, 600.0, 600.0, 600.0, 20.0]),
        "D": np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0]),
    }
    # 精密模式（reset/抓取移动时使用，高刚度抗下垂）
    PRECISION_PARAM = {
        "K": np.array([4000.0, 4000.0, 4000.0, 600.0, 600.0, 600.0, 20.0]),
        "D": np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0]),
    }

    # ==================== 关节阻抗参数 ====================
    # 用于IMPEDANCE_MODE="joint"时的step控制
    # 关节阻抗需要低刚度配低阻尼，避免震动，实现柔顺控制
    # K: 关节刚度 (N·m/度), D: 关节阻尼 (N·m/(度/秒)), 范围0-1
    JOINT_COMPLIANCE_PARAM = {
        "K": np.array([12.0, 12.0, 12.0, 10.0, 9.0, 9.0, 7.0]),  # 官方推荐值
        "D": np.array([0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]),
    }

    # ==================== 控制参数 ====================
    MAX_EPISODE_LENGTH = 200
    HZ = 10
    VEL_RATIO = 50
    ACC_RATIO = 50

    # USB 复位模式
    AUTO_RESET_USB = False
    MANUAL_RESET_TIMEOUT = 8.0

    # 夹爪配置
    USE_GRIPPER = True
    GRIPPER_MOTOR_ID = 1
    GRIPPER_SLEEP = 0.2  # 夹爪动作后等待时间（秒）- 优化：减少连续开闭延迟


class TrainConfig(DefaultTrainingConfig):
    """训练配置"""

    # ==================== 环境设置 ====================
    image_keys = ["side_policy","wrist_1", "wrist_2"]  # 策略观测使用的图像
    classifier_keys = ["side_classifier"]  # 分类器使用的图像

    # Proprio状态键 (修改：添加力/力矩观测，匹配Franka配置)
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose", "last_action"]

    # ==================== 人工标注配置 ====================
    ENABLE_SPACE_KEY = False  # True=启用空格键, False=仅脚踏板

    # ==================== 训练参数 ====================
    # 内存优化：降低 batch_size 和 cta_ratio 减少训练缓存
    batch_size = 256  # 从 256 → 128 (节省 ~5GB)
    max_steps = 1_000_000
    random_steps = 500

    # ==================== Replay Buffer 配置 ====================
    # ⚠️ np.empty() 会在初始化时立即分配全部内存 (即使数据还未填充)
    # 系统总内存60GB，训练进程需控制在30GB以内:
    #   - Buffer × 2 (replay + demo): 5000 × 2 × 2 × 0.56MB = ~11.2GB
    #   - 网络 + 训练缓存 (batch=128, cta=2): ~8GB
    #   - XLA 编译缓存: ~5GB
    #   - 图像解码缓存: ~5GB
    #   - 系统 + 其他: ~10GB
    # 总计: ~39GB → 需要进一步优化
    replay_buffer_capacity = 200000  # 从默认200000降低到10000 (足够训练)

    # ==================== 算法参数 ====================
    agent = "drq"  # 使用DrQ算法
    encoder_type = "resnet-pretrained"
    demo_path = "/home/xlb/code_marvin/hil-serl/examples/experiments/resnet10_params.pkl"  # Demo数据路径（可选）

    # ==================== 奖励配置 ====================
    checkpoint_period = 2000  # 模型checkpoint保存周期
    cta_ratio = 2  # critic-to-actor更新比例 (保持2，已经很低)
    discount = 0.98  # 折扣因子
    buffer_period = 1000  # buffer保存周期
    setup_mode = "single-arm-learned-gripper"  # 设置模式

    def get_environment(self, fake_env=False, save_video=False, classifier=False, collect_classifier_data=False):
        """
        创建Marvin USB任务环境

        Args:
            fake_env: 是否为虚拟环境（测试用）
            save_video: 是否保存视频
            classifier: 是否使用learned reward（加载分类器wrapper）
            collect_classifier_data: 是否采集分类器数据（使用classifier图像但不加载wrapper）

        Returns:
            配置好的环境
        """
        from marvin_env.envs.marvin_env import MarvinEnv
        from examples.experiments.marvin_usb_insertion.wrapper import MarvinUSBEnv, GripperPenaltyWrapper
        from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ImageFilterWrapper, StateNormalizationWrapper
        from serl_launcher.wrappers.chunking import ChunkingWrapper
        from serl_launcher.networks.reward_classifier import load_classifier_func
        from franka_env.envs.relative_env import RelativeFrame
        from franka_env.envs.wrappers import (
            Quat2EulerWrapper,
            SpacemouseIntervention,
            MultiCameraBinaryRewardClassifierWrapper,
        )

        # 创建基础环境
        env_config = MarvinUSBEnvConfig()
        env = MarvinUSBEnv(
            hz=env_config.HZ,
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )

        # 应用标准Wrapper（与原始顺序一致）
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)

        # 夹爪惩罚（防止频繁夹爪动作）
        if env_config.USE_GRIPPER:
            env = GripperPenaltyWrapper(env, penalty=-0.05)

        # 图像过滤（根据模式选择图像）
        # collect_classifier_data=True: 仅采集分类器数据，使用classifier_keys，不加载wrapper
        # classifier=True: RL训练，同时需要policy和classifier图像
        # 否则: 使用image_keys（策略训练without learned reward）
        if collect_classifier_data:
            # 仅采集分类器数据：只需要classifier图像
            image_keys = self.classifier_keys
        elif classifier:
            # RL训练with learned reward：需要所有图像（policy + classifier）
            all_image_keys = list(set(self.image_keys + self.classifier_keys))
            image_keys = all_image_keys
        else:
            # 普通策略训练：只需要policy图像
            image_keys = self.image_keys
        env = ImageFilterWrapper(env, image_keys=image_keys)

        # 状态归一化（在 SERLObsWrapper 之前，处理字典state）
        env = StateNormalizationWrapper(env, normalization_params=env_config.STATE_NORMALIZATION)

        # 观测包装（将state扁平化，图像保持不变）
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # Chunking（动作序列）
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 奖励分类器（仅在classifier=True时使用learned reward）
        # 注意：collect_classifier_data=True时不加载分类器wrapper（避免自动done）
        if classifier:
            classifier_func = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.join(examples_path, "classifier_ckpt/checkpoint_150"),
            )

            # 脚踏板监听：human labeling of success
            from pynput import keyboard as pynput_keyboard

            _space_pressed = [False]  # mutability hack for closure
            _last_action = [None]  # 用于动作平滑度惩罚
            _shift_pressed = [False]  # 追踪 Shift 键状态

            def _on_press(key):
                try:
                    # 追踪 Shift 键
                    if key in [pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r]:
                        _shift_pressed[0] = True

                    # 脚踏板1: Shift + 左方向键 = 成功
                    if key == pynput_keyboard.Key.right and _shift_pressed[0]:
                        _space_pressed[0] = True
                        print("\n🦶 [PEDAL-1] 脚踏板1按下！标记为成功 (Shift+←)\n")

                    # 空格键支持（受配置控制）
                    elif key == pynput_keyboard.Key.space and self.ENABLE_SPACE_KEY:
                        _space_pressed[0] = True
                        print("\n🎯 [SPACE KEY] 空格键按下！标记为成功\n")
                except Exception as e:
                    print(f"[KEY ERROR] {e}")

            def _on_release(key):
                try:
                    # 释放 Shift 键
                    if key in [pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_l, pynput_keyboard.Key.shift_r]:
                        _shift_pressed[0] = False
                except Exception:
                    pass

            _listener = pynput_keyboard.Listener(on_press=_on_press, on_release=_on_release)
            _listener.daemon = True
            _listener.start()
            print("[PEDAL] 监听器已启动")
            print("[PEDAL] 🦶 踩脚踏板1 (Shift+←) 标记成功")
            if self.ENABLE_SPACE_KEY:
                print("[PEDAL] ⌨️  空格键备选已启用")
            else:
                print("[PEDAL] ⌨️  空格键已禁用（ENABLE_SPACE_KEY=False）")

            def reward_func(obs):
                """
                组合奖励函数：
                1. 分类器预测插入成功（sigmoid > 0.8） + 夹爪松开
                2. 脚踏板人工标注成功（Shift+←）
                3. 动作平滑度惩罚（减少抖动）
                """
                # 脚踏板人工标注
                if _space_pressed[0]:
                    _space_pressed[0] = False
                    print("\n✅ [REWARD] 脚踏板触发，返回奖励=1.0\n")
                    return 1.0

                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))

                # ==================== 完整调试信息 ====================
                print("\n" + "=" * 70)
                print("[REWARD_DEBUG] 观测空间详细分析")
                print("=" * 70)

                # 1. 观测结构
                print(f"[REWARD_DEBUG] obs keys: {obs.keys()}")
                print(f"[REWARD_DEBUG] obs['state'] shape: {obs['state'].shape}")
                print(f"[REWARD_DEBUG] obs['state'] dtype: {obs['state'].dtype}")

                # 2. 展开观测数据（去除时间维度）
                state_flat = obs["state"][0]  # shape: [state_dim]
                print(f"[REWARD_DEBUG] state_flat shape: {state_flat.shape}")
                print(f"[REWARD_DEBUG] state_flat length: {len(state_flat)}")
                print(f"[REWARD_DEBUG] state_flat 完整值:\n  {state_flat}")

                # 3. 按维度拆解（期望24维）
                # ⚠️ 注意：gymnasium的flatten()函数会按照字母顺序排序keys！
                # 实际顺序: gripper_pose(1) + tcp_force(3) + tcp_pose(6) + tcp_torque(3) + tcp_vel(6) + last_action(5) = 24维
                # 不是我们指定的: tcp_pose + tcp_vel + tcp_force + tcp_torque + gripper_pose + last_action
                print(f"\n[REWARD_DEBUG] 维度拆解 (期望24维，按字母序):")

                if len(state_flat) >= 1:
                    gripper = state_flat[0]
                    print(f"  [0]      gripper_pose (1维): {gripper:.4f}")
                else:
                    print(f"  [0]      gripper_pose: ❌ 缺失")

                if len(state_flat) >= 4:
                    tcp_force = state_flat[1:4]
                    print(f"  [1:4]    tcp_force (3维): [{tcp_force[0]:.3f}, {tcp_force[1]:.3f}, {tcp_force[2]:.3f}]")
                else:
                    print(f"  [1:4]    tcp_force: ❌ 缺失")

                if len(state_flat) >= 10:
                    tcp_pose = state_flat[4:10]
                    print(f"  [4:10]   tcp_pose (6维): [{', '.join([f'{x:.4f}' for x in tcp_pose])}]")
                    print(f"           位置XYZ(m): [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}]")
                    print(f"           位置XYZ(mm): [{tcp_pose[0]*1000:.1f}, {tcp_pose[1]*1000:.1f}, {tcp_pose[2]*1000:.1f}]")
                    print(f"           欧拉角(rad): [{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}]")
                    print(f"           欧拉角(度): [{jnp.rad2deg(tcp_pose[3]):.2f}, {jnp.rad2deg(tcp_pose[4]):.2f}, {jnp.rad2deg(tcp_pose[5]):.2f}]")
                else:
                    print(f"  [4:10]   tcp_pose: ❌ 缺失")

                if len(state_flat) >= 13:
                    tcp_torque = state_flat[10:13]
                    print(f"  [10:13]  tcp_torque (3维): [{tcp_torque[0]:.3f}, {tcp_torque[1]:.3f}, {tcp_torque[2]:.3f}]")
                else:
                    print(f"  [10:13]  tcp_torque: ❌ 缺失")

                if len(state_flat) >= 19:
                    tcp_vel = state_flat[13:19]
                    print(f"  [13:19]  tcp_vel (6维): [{', '.join([f'{x:.4f}' for x in tcp_vel])}]")
                    print(f"           线速度xyz(m/s): [{tcp_vel[0]:.4f}, {tcp_vel[1]:.4f}, {tcp_vel[2]:.4f}]")
                    print(f"           角速度xyz(rad/s): [{tcp_vel[3]:.4f}, {tcp_vel[4]:.4f}, {tcp_vel[5]:.4f}]")
                else:
                    print(f"  [13:19]  tcp_vel: ❌ 缺失")

                if len(state_flat) >= 24:
                    last_action = state_flat[19:24]
                    print(f"  [19:24]  last_action (5维): [{', '.join([f'{x:.4f}' for x in last_action])}]")
                    print(f"           [dx={last_action[0]:.4f}, dy={last_action[1]:.4f}, dz={last_action[2]:.4f}, drz={last_action[3]:.4f}, gripper={last_action[4]:.4f}]")
                else:
                    print(f"  [19:24]  last_action: ❌ 缺失")

                # 4. 获取分类器输出
                classifier_logit = classifier_func(obs)
                sigmoid_val = sigmoid(classifier_logit)
                print(f"\n[REWARD_DEBUG] 分类器输出:")
                print(f"  logit: {classifier_logit}")
                print(f"  sigmoid: {sigmoid_val}")

                # 5. 提取夹爪位置（根据实际维度）
                # ⚠️ 注意：flatten按字母序排列，gripper_pose在索引[0]，不是[18]！
                # 实际顺序: gripper_pose(1) + tcp_force(3) + tcp_pose(6) + tcp_torque(3) + tcp_vel(6)
                if len(state_flat) >= 1:
                    gripper_pose_val = float(state_flat[0])  # 字母序第一个
                else:
                    gripper_pose_val = 0.0

                print(f"\n[REWARD_DEBUG] 奖励判断:")
                print(f"  gripper_pose: {gripper_pose_val:.3f}")

                # 计算基础奖励
                classifier_success = float(sigmoid_val.item()) > 1.70  # 1.7禁用分类器，只用空格键监听
                gripper_opened = gripper_pose_val > 0.5
                base_reward = int(classifier_success and gripper_opened)

                # 动作平滑度惩罚
                smoothness_penalty = 0.0
                if len(state_flat) >= 24:
                    # last_action 在字母序最后: gripper(1) + tcp_force(3) + tcp_pose(6) + tcp_torque(3) + tcp_vel(6) + last_action(5) = 24维
                    last_action = state_flat[19:24]  # [dx, dy, dz, drz, gripper]
                    if _last_action[0] is not None:
                        action_diff = jnp.array(last_action) - jnp.array(_last_action[0])
                        action_diff_norm = float(jnp.sqrt(jnp.sum(action_diff ** 2)))
                        smoothness_penalty = 0.001 * action_diff_norm
                        print(f"  last_action (t-1): [{', '.join([f'{x:.4f}' for x in _last_action[0]])}]")
                        print(f"  last_action (t):   [{', '.join([f'{x:.4f}' for x in last_action])}]")
                        print(f"  action_diff: [{', '.join([f'{x:.4f}' for x in action_diff])}]")
                        print(f"  action_diff_norm: {action_diff_norm:.6f}")
                    else:
                        print(f"  last_action (t-1): None (首步)")
                        print(f"  last_action (t):   [{', '.join([f'{x:.4f}' for x in last_action])}]")
                    _last_action[0] = last_action.copy()
                else:
                    print(f"  ❌ state_flat 维度不足24，无法提取 last_action")

                # 应用平滑度惩罚
                # wrapper 已修复: done = done or (rew >= 0.5)，支持负奖励
                final_reward = base_reward - smoothness_penalty

                print(f"  classifier_success (>0.70): {classifier_success}")
                print(f"  gripper_opened (>0.5): {gripper_opened}")
                print(f"  base_reward: {base_reward}")
                print(f"  smoothness_penalty: {smoothness_penalty:.6f}")
                print(f"  final_reward: {final_reward:.6f}")
                print("=" * 70 + "\n")

                return final_reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
            env = SpacemouseIntervention(env)

        # SpaceMouse人类干预（仅训练时，在最后应用）
        if not classifier:
            env = SpacemouseIntervention(env)

        return env
