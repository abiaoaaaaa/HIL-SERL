import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
import numpy as np


class SERLObsWrapper(gym.ObservationWrapper):
    """
    SERL观测空间包装器

    功能：
    1. 从state字典中选择指定的proprio_keys
    2. 将选中的状态扁平化为一维向量
    3. 与图像一起组成最终观测

    示例：
        输入观测 = {
            "state": {
                "tcp_pose": [7维],
                "tcp_vel": [6维],
                "tcp_force": [3维],  # 可选
                "gripper_pose": [1维]
            },
            "images": {"camera_1": [128,128,3]}
        }

        选择 proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]

        输出观测 = {
            "state": [14维扁平向量],  # 7+6+1
            "camera_1": [128,128,3]
        }
    """

    def __init__(self, env, proprio_keys=None):
        """
        Args:
            env: 被包装的环境
            proprio_keys: 选择哪些状态键（None表示全选）
                         常用: ["tcp_pose", "tcp_vel", "gripper_pose"]
                         可选添加: "tcp_force", "tcp_torque"
        """
        super().__init__(env)
        self.proprio_keys = proprio_keys

        # 如果未指定，使用所有可用的状态键
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        # 构建选中状态的空间
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # 定义新的观测空间：扁平化的状态 + 图像
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),  # 扁平化为一维向量
                **(self.env.observation_space["images"]),     # 保持图像不变
            }
        )

    def observation(self, obs):
        """
        转换观测格式

        将字典形式的状态扁平化为向量
        """
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),  # 图像直接展开
        }
        return obs

    def reset(self, **kwargs):
        """重置时也要转换观测"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
        }
        return obs


class ImageFilterWrapper(gym.ObservationWrapper):
    """
    图像过滤包装器

    功能：从观测中选择指定的图像键

    示例：
        输入观测 = {
            "state": {...},
            "images": {
                "wrist_1": [128,128,3],
                "wrist_2": [128,128,3],
                "side_policy": [128,128,3],
                "side_classifier": [128,128,3]
            }
        }

        选择 image_keys = ["wrist_1", "side_policy"]

        输出观测 = {
            "state": {...},
            "images": {
                "wrist_1": [128,128,3],
                "side_policy": [128,128,3]
            }
        }
    """

    def __init__(self, env, image_keys=None):
        """
        Args:
            env: 被包装的环境
            image_keys: 要保留的图像键列表（None表示保留所有）
        """
        super().__init__(env)

        # 如果未指定，保留所有图像
        if image_keys is None:
            self.image_keys = list(self.env.observation_space["images"].keys())
        else:
            self.image_keys = image_keys
            # 验证所有指定的键都存在
            available_keys = set(self.env.observation_space["images"].keys())
            requested_keys = set(image_keys)
            if not requested_keys.issubset(available_keys):
                missing = requested_keys - available_keys
                raise ValueError(
                    f"Requested image keys {missing} not found in observation space. "
                    f"Available keys: {available_keys}"
                )

        # 构建新的观测空间：只包含选中的图像
        self.observation_space = gym.spaces.Dict(
            {
                "state": self.env.observation_space["state"],
                "images": gym.spaces.Dict(
                    {key: self.env.observation_space["images"][key]
                     for key in self.image_keys}
                ),
            }
        )

    def observation(self, obs):
        """过滤观测中的图像"""
        return {
            "state": obs["state"],
            "images": {key: obs["images"][key] for key in self.image_keys},
        }

    def reset(self, **kwargs):
        """重置时也要过滤观测"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class StateNormalizationWrapper(gym.ObservationWrapper):
    """
    状态归一化包装器

    功能：
    1. 对字典形式的state中每个键进行独立归一化
    2. 使用 (value - mean) / std 公式
    3. 图像保持不变

    注意：
    - 应该在 SERLObsWrapper **之前**使用（处理字典state）
    - 或者在 SERLObsWrapper **之后**处理扁平化的state向量

    示例（SERLObsWrapper之前）：
        输入观测 = {
            "state": {
                "tcp_pose": [7维],      # 原始值: [0.4m, 0.3m, ...]
                "tcp_force": [3维],     # 原始值: [15N, -8N, 3N]
            },
            "images": {...}
        }

        归一化参数 = {
            "tcp_pose": {"mean": [0.4,...], "std": [0.15,...]},
            "tcp_force": {"mean": [0,0,0], "std": [10,10,10]},
        }

        输出观测 = {
            "state": {
                "tcp_pose": [0, -0.67, ...],    # 归一化后
                "tcp_force": [1.5, -0.8, 0.3],  # 归一化后
            },
            "images": {...}  # 保持不变
        }
    """

    def __init__(self, env, normalization_params):
        """
        Args:
            env: 被包装的环境
            normalization_params: 归一化参数字典
                格式: {
                    "state_key": {
                        "mean": [array],
                        "std": [array]
                    },
                    ...
                }
        """
        super().__init__(env)
        self.normalization_params = normalization_params

        # 转换为numpy数组以提高性能
        self.means = {}
        self.stds = {}
        for key, params in normalization_params.items():
            self.means[key] = np.array(params["mean"], dtype=np.float32)
            self.stds[key] = np.array(params["std"], dtype=np.float32)

        # 观测空间保持不变（归一化不改变形状）
        self.observation_space = env.observation_space

    def observation(self, obs):
        """
        对观测中的state进行归一化

        归一化公式: normalized = (raw - mean) / std
        """
        # 深拷贝避免修改原始观测
        normalized_obs = {}

        # 处理state字典
        if "state" in obs:
            normalized_state = {}
            for key, value in obs["state"].items():
                if key in self.means:
                    # 归一化: (value - mean) / std
                    normalized_value = (value - self.means[key]) / self.stds[key]
                    normalized_state[key] = normalized_value.astype(np.float32)
                else:
                    # 如果没有归一化参数，保持原值
                    normalized_state[key] = value
            normalized_obs["state"] = normalized_state

        # 图像保持不变
        if "images" in obs:
            normalized_obs["images"] = obs["images"]

        return normalized_obs

    def reset(self, **kwargs):
        """重置时也要归一化观测"""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info