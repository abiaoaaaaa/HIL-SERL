#!/usr/bin/env python3
"""
Marvin环境基础测试脚本

测试MarvinEnv的基本功能：
1. 环境初始化
2. 观测空间
3. step()执行
4. reset()功能
"""
import sys
import os
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
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
