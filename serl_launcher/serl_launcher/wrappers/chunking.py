from collections import deque
from typing import Optional

import gymnasium as gym
import gymnasium.spaces
import jax
import numpy as np


def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree.map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: isinstance(x, list)
    )


def space_stack(space: gym.Space, repeat: int):
    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise TypeError()


class ChunkingWrapper(gym.Wrapper):
    """
    观测历史与动作序列包装器

    功能1: 观测历史 (Observation History)
    - 累积过去N步的观测，形成历史序列
    - 用于时序建模（LSTM/Transformer）

    功能2: 动作序列 (Action Chunking / Receding Horizon Control)
    - 一次预测多步动作，然后逐步执行
    - 提高动作平滑性和一致性

    示例：
        obs_horizon = 3, act_exec_horizon = 2

        策略输入: 最近3步观测 [obs_t-2, obs_t-1, obs_t]
        策略输出: 未来2步动作 [act_t, act_t+1]
        环境执行: 先执行act_t，再执行act_t+1

    常用配置：
        obs_horizon=1, act_exec_horizon=None  # 标准单步控制
        obs_horizon=3, act_exec_horizon=4     # ACT风格动作chunking
    """

    def __init__(self, env: gym.Env, obs_horizon: int, act_exec_horizon: Optional[int]):
        """
        Args:
            env: 被包装的环境
            obs_horizon: 观测历史长度
                        1 = 只用当前观测（无历史）
                        >1 = 累积多步观测
            act_exec_horizon: 动作序列长度
                             None = 单步动作（标准RL）
                             >1 = 一次预测多步动作
        """
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.act_exec_horizon = act_exec_horizon

        # 使用deque维护观测历史（自动淘汰最旧的）
        self.current_obs = deque(maxlen=self.obs_horizon)

        # 扩展观测空间：堆叠obs_horizon个观测
        self.observation_space = space_stack(
            self.env.observation_space, self.obs_horizon
        )

        # 扩展动作空间（如果使用动作序列）
        if self.act_exec_horizon is None:
            self.action_space = self.env.action_space  # 单步动作
        else:
            self.action_space = space_stack(
                self.env.action_space, self.act_exec_horizon  # 多步动作
            )

    def step(self, action, *args):
        """
        执行动作序列

        流程：
        1. 如果是单步动作，包装成列表
        2. 逐步执行每个动作
        3. 累积观测到历史队列
        4. 返回堆叠的观测历史

        注意：只返回最后一步的reward、done、info
        """
        act_exec_horizon = self.act_exec_horizon
        if act_exec_horizon is None:
            action = [action]  # 单步动作包装成列表
            act_exec_horizon = 1

        assert len(action) >= act_exec_horizon

        # 逐步执行动作序列
        for i in range(act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
            self.current_obs.append(obs)  # 累积观测

        # 返回堆叠的观测历史 [obs_t-N+1, ..., obs_t]
        return (stack_obs(self.current_obs), reward, done, trunc, info)

    def reset(self, **kwargs):
        """
        重置时初始化观测历史

        策略：用第一个观测填充整个历史
        例如: obs_horizon=3 → [obs_0, obs_0, obs_0]
        """
        obs, info = self.env.reset(**kwargs)
        # 用初始观测填充历史队列
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info


def post_stack_obs(obs, obs_horizon=1):
    if obs_horizon != 1:
        # TODO: Support proper stacking
        raise NotImplementedError("Only obs_horizon=1 is supported for now")
    obs = {k: v[None] for k, v in obs.items()}
    return obs