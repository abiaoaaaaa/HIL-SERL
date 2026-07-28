"""
实现 ResNet 特征预编码的完整改动清单

背景：
- 当前：Buffer 存原始图像 (600KB/transition × 10000 = 6GB)
- 目标：Buffer 存 ResNet 特征 (6KB/transition × 10000 = 60MB)
- 节省：99% 内存

⚠️ 预计工作量：2-3天，需要深入理解 JAX/Flax 和 SERL 架构

================================================================================
步骤 1: 创建特征编码 Wrapper (✅ 已完成)
================================================================================
位置: serl_robot_infra/franka_env/envs/feature_encoding_wrapper.py

关键点:
- 在 env.step() 后立即编码图像
- 修改 observation_space: {"image": Box(H,W,C)} → {"image_features": Box(feature_dim,)}


================================================================================
步骤 2: 修改 Config (✅ 已完成)
================================================================================
位置: examples/experiments/marvin_usb_insertion/config.py:222-235

新增配置:
  use_feature_encoding = True
  feature_encoder_dim = 512


================================================================================
步骤 3: 修改 get_environment() 应用 Wrapper (⚠️ 待做)
================================================================================
位置: examples/experiments/marvin_usb_insertion/config.py:270-340

在 get_environment() 中加入:

```python
if self.use_feature_encoding:
    # 1. 初始化预训练 ResNet encoder
    from serl_launcher.vision.resnet_v1 import resnetv1_configs
    import jax
    
    rng = jax.random.PRNGKey(0)
    encoder_def = resnetv1_configs["resnetv1-10-frozen"](
        pre_pooling=True,  # 输出 (H', W', C') 而非池化后的向量
        name="pretrained_encoder",
    )
    
    # 加载预训练权重
    encoder_params = ... # 从 demo_path 加载
    
    # 创建编码函数
    @jax.jit
    def encode_fn(img):
        features = encoder_def.apply({"params": encoder_params}, img, train=False)
        # 池化: (H', W', C') → (feature_dim,)
        return features.mean(axis=(1, 2))  # avg pooling
    
    # 2. 应用 Wrapper
    from franka_env.envs.feature_encoding_wrapper import FeatureEncodingWrapper
    env = FeatureEncodingWrapper(
        env,
        encoder_fn=encode_fn,
        image_keys=tuple(self.image_keys),
        feature_dim=self.feature_encoder_dim,
    )
```


================================================================================
步骤 4: 修改 Agent 网络架构跳过编码 (⚠️ 待做，最复杂)
================================================================================
位置: serl_launcher/agents/continuous/sac.py:437-500

问题: 现在 agent 网络期望输入图像 → 内部编码 → 特征 → Q/policy
目标: agent 网络直接接收特征 → Q/policy (跳过编码层)

方案 A: 新增 encoder_type="identity"
```python
def make_agent(
    ...
    encoder_type: str = "resnet-pretrained",  # 新增 "identity" 选项
):
    if encoder_type == "identity":
        # 直接使用特征，不编码
        encoders = {
            image_key: nn.Sequential([])  # 空操作
            for image_key in image_keys
        }
    elif encoder_type == "resnet-pretrained":
        # 原有逻辑...
```

方案 B (更干净): 修改观测空间，移除图像key
- agent 直接从 obs["state"] 读取拼接好的特征
- 需要在 Wrapper 中把 `image_features` 拼接到 `state`


================================================================================
步骤 5: 验证 ReplayBuffer 兼容性 (⚠️ 待做)
================================================================================
位置: serl_launcher/data/replay_buffer.py

检查点:
1. 观测空间改变后 Buffer 能否正确初始化
2. 特征向量 dtype=float32 是否兼容 (原图像 uint8)
3. sample() 返回的数据形状是否正确

测试脚本:
```python
from gym import spaces
import numpy as np

obs_space = spaces.Dict({
    "wrist_1_features": spaces.Box(-np.inf, np.inf, (512,), np.float32),
    "state": spaces.Box(-np.inf, np.inf, (24,), np.float32),
})
action_space = spaces.Box(-1, 1, (5,), np.float32)

buffer = ReplayBuffer(obs_space, action_space, capacity=100)
# 测试 insert/sample...
```


================================================================================
步骤 6: Actor 部署时的编码 (⚠️ 待做)
================================================================================
问题: Actor 部署时从真实环境获取原始图像，但 agent 期望特征

方案 A: Actor 端也用 FeatureEncodingWrapper
- 优点: 代码统一
- 缺点: Actor 需要加载 encoder 权重 (增加启动时间)

方案 B: Actor 端手动编码
```python
# 在 actor.py 的 select_action() 中
if config.use_feature_encoding:
    for key in image_keys:
        obs[key + "_features"] = self.encoder_fn(obs.pop(key))
```


================================================================================
步骤 7: 完整测试流程
================================================================================
1. 单元测试: Wrapper 输出形状正确
2. Buffer 测试: 存取特征无误
3. Agent 测试: 网络前向传播正确
4. 集成测试: Learner 训练正常
5. Actor 测试: 推理正常
6. 内存测试: top/htop 监控实际内存占用


================================================================================
估算工作量
================================================================================
- 步骤 3: 30分钟
- 步骤 4: 4-6小时 (需要理解 JAX/Flax 网络架构)
- 步骤 5: 1-2小时
- 步骤 6: 1-2小时
- 步骤 7: 2-4小时 (调试)
总计: 1-2天开发 + 1天调试


================================================================================
潜在风险
================================================================================
1. **训练效果变差**: 预编码的特征是固定的，不能端到端微调
2. **分布偏移**: 预训练 ResNet 在 ImageNet 上训练，可能不适配机器人图像
3. **信息丢失**: 池化后的特征可能丢失空间细节
4. **调试困难**: 特征不可视化，难以排查问题


================================================================================
替代方案 (更简单)
================================================================================
1. **降低图像分辨率**: 256×256 → 128×128 (节省 75% 内存)
2. **减少相机数量**: 4 → 2 (节省 50% 内存)
3. **使用 JPEG 压缩**: 存储时压缩，训练时解压 (节省 90%，但增加 CPU)
4. **增加物理内存**: 60GB → 128GB (一劳永逸)
"""