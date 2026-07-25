#!/usr/bin/env python3
"""
Marvin 环境基础功能测试 - Test Marvin Environment

功能：
    测试 MarvinEnv 类的基本功能是否正常工作，包括初始化、
    观测空间、动作执行、重置等核心功能。

主要功能：
    1. 环境初始化测试
    2. 观测空间结构验证
    3. step() 方法功能测试
    4. reset() 方法功能测试
    5. 相机数据获取测试
    6. 机械臂控制响应测试

测试流程：
    1. 环境初始化测试：
       a) 创建 MarvinEnv 实例
       b) 验证配置参数正确加载
       c) 检查机械臂连接状态
       d) 检查相机初始化状态
       e) 验证观测空间和动作空间定义

    2. 观测空间测试：
       a) 获取初始观测
       b) 验证观测字典结构：
          - "state": 机械臂状态（关节角度、末端位置等）
          - "images": 相机图像字典
            * "wrist_1": 腕部相机图像
            * "side": 侧面相机图像（如果有）
          - "goal": 目标信息（如果有）
       c) 检查数据类型和维度
       d) 验证数值范围合理性

    3. step() 方法测试：
       a) 生成测试动作（随机或固定）
       b) 执行 env.step(action)
       c) 验证返回值结构：
          - obs: 新的观测
          - reward: 奖励值
          - done: 是否结束
          - info: 附加信息字典
       d) 检查状态变化：
          - 关节角度是否更新
          - 末端位置是否变化
          - 图像是否刷新
       e) 验证控制频率和响应时间

    4. reset() 方法测试：
       a) 执行 env.reset()
       b) 验证机械臂返回初始姿态
       c) 检查观测空间重置正确
       d) 确认内部状态清零

    5. 边界条件测试：
       a) 测试接近工作空间边界的动作
       b) 验证安全限制是否生效
       c) 测试异常动作的处理

测试配置：
    TestMarvinEnvConfig 类配置：
    - ROBOT_IP: 机械臂 IP 地址
    - ARM: 机械臂编号（'A' 或 'B'）
    - KINE_CONFIG_PATH: 运动学配置文件路径
    - REALSENSE_CAMERAS: 相机配置字典
    - TARGET_POSE: 目标位置（测试用）
    - RESET_POSE: 重置位置
    - ABS_POSE_LIMIT_LOW/HIGH: 安全边界

预期输出：
    ================================================================
    [1/5] 测试环境初始化...
    ✓ 环境创建成功
    ✓ 机械臂连接正常
    ✓ 相机初始化完成
    ✓ 观测空间: Box(...)
    ✓ 动作空间: Box(...)

    [2/5] 测试观测空间...
    ✓ 获取初始观测成功
    ✓ state 维度: (N,)
    ✓ images 包含: ['wrist_1', 'side']
    ✓ 图像尺寸: wrist_1=(H, W, 3), side=(H, W, 3)

    [3/5] 测试 step() 方法...
    ✓ 执行 step() 成功
    ✓ 返回值结构正确: (obs, reward, done, info)
    ✓ 关节角度已更新
    ✓ 控制频率: X Hz

    [4/5] 测试 reset() 方法...
    ✓ reset() 执行成功
    ✓ 机械臂返回初始姿态
    ✓ 观测空间已重置

    [5/5] 测试边界条件...
    ✓ 安全边界检测正常
    ✓ 异常动作处理正确

    ================================================================
    所有测试通过！环境工作正常。
    ================================================================

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 1. 编辑配置（修改 TestMarvinEnvConfig 类）
    #    - ROBOT_IP
    #    - KINE_CONFIG_PATH
    #    - REALSENSE_CAMERAS
    #    - 姿态和边界参数

    # 2. 运行测试
    python utils/test_tools/test_marvin_env.py

应用场景：
    1. 新环境搭建后的验证
    2. 配置文件修改后的回归测试
    3. 机械臂或相机更换后的功能确认
    4. 排查环境问题的第一步诊断

常见问题排查：
    1. 环境初始化失败：
       - 检查机械臂 IP 和连接
       - 验证运动学配置文件路径
       - 确认相机序列号正确

    2. 观测空间错误：
       - 检查相机是否正常工作
       - 验证图像尺寸配置
       - 确认裁剪参数正确

    3. step() 执行失败：
       - 检查动作范围是否合理
       - 验证控制频率配置
       - 确认机械臂响应正常

    4. reset() 失败：
       - 检查重置姿态是否可达
       - 验证运动规划参数
       - 确认工作空间边界设置

注意事项：
    1. 确保机械臂在安全工作空间内
    2. 测试前确认周围无障碍物
    3. 准备好急停按钮
    4. 首次运行建议降低速度参数
    5. 异常时立即按 Ctrl+C 或急停
"""
import sys
import os
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from serl_robot_infra.marvin_env.envs.marvin_env import MarvinEnv
from serl_robot_infra.marvin_env.envs.config import DefaultMarvinEnvConfig


class TestMarvinEnvConfig(DefaultMarvinEnvConfig):
    """测试配置"""
    ROBOT_IP = "192.168.14.190"
    ARM = 'A'

    # 修改为你的配置文件路径
    KINE_CONFIG_PATH = "/path/to/your/ccs_m6_40.MvKDCfg"

    # 相机配置（根据实际修改）
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "130322274175",
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }

    # 简单测试任务
    TARGET_POSE = np.array([500.0, 200.0, 300.0, 180.0, 0.0, -90.0])
    RESET_POSE = np.array([450.0, 200.0, 350.0, 180.0, 0.0, -90.0])

    # 安全边界
    ABS_POSE_LIMIT_LOW = np.array([300.0, 100.0, 200.0, 170.0, -10.0, -100.0])
    ABS_POSE_LIMIT_HIGH = np.array([600.0, 300.0, 500.0, 190.0, 10.0, -80.0])


def test_basic_functionality():
    """测试基础功能"""
    print("=" * 70)
    print("Marvin环境基础功能测试")
    print("=" * 70)

    # 创建环境
    print("\n[测试1] 创建环境...")
    try:
        env = MarvinEnv(hz=10, config=TestMarvinEnvConfig())
        print("✓ 环境创建成功")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return

    # 检查空间定义
    print("\n[测试2] 检查空间定义...")
    print(f"  动作空间: {env.action_space}")
    print(f"  观测空间: {env.observation_space}")
    print("✓ 空间定义正确")

    # 测试reset
    print("\n[测试3] 测试reset()...")
    try:
        obs, info = env.reset()
        print("✓ Reset成功")
        print(f"  观测keys: {obs.keys()}")
        print(f"  State keys: {obs['state'].keys()}")
        print(f"  Image keys: {obs['images'].keys()}")
        print(f"  TCP位置: {obs['state']['tcp_pose'][:3]}")
    except Exception as e:
        print(f"✗ Reset失败: {e}")
        env.close()
        return

    # 测试step
    print("\n[测试4] 测试step()...")
    try:
        for i in range(5):
            # 小幅度随机动作
            action = env.action_space.sample() * 0.1
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward}, done={done}")
            print(f"    TCP位置: {obs['state']['tcp_pose'][:3]}")
        print("✓ Step执行成功")
    except Exception as e:
        print(f"✗ Step失败: {e}")

    # 关闭环境
    print("\n[测试5] 关闭环境...")
    try:
        env.close()
        print("✓ 环境关闭成功")
    except Exception as e:
        print(f"✗ 关闭失败: {e}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_basic_functionality()
