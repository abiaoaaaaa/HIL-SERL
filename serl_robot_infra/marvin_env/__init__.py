"""
Marvin机械臂环境模块

提供与HIL-SERL框架兼容的Marvin机器人Gym环境接口
"""

from marvin_env.envs.marvin_env import MarvinEnv
from marvin_env.envs.config import DefaultMarvinEnvConfig

__all__ = ['MarvinEnv', 'DefaultMarvinEnvConfig']
