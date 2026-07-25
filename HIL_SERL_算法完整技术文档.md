# HIL-SERL 算法完整技术文档

## 目录
1. [算法概述](#1-算法概述)
2. [网络结构与数据流](#2-网络结构与数据流)
3. [训练流程](#3-训练流程)
4. [参数更新机制](#4-参数更新机制)
5. [关键超参数](#5-关键超参数)

---

## 1. 算法概述

### 1.1 算法定位
- **名称**: HIL-SERL (Human-in-the-Loop SERL)
- **基础算法**: SAC (Soft Actor-Critic) + RLPD (RL with Prior Data)
- **适用场景**: 双臂机械臂精细操作任务
- **核心特色**: 
  - 混合动作空间 (连续末端执行器 + 离散夹爪动作)
  - 在线人类干预机制 (SpaceMouse)
  - 50/50 混合采样 (演示数据 + 在线数据)
  - 多相机视觉输入 + 本体感知融合
  - 分布式 Actor-Learner 架构

### 1.2 算法组成
```
HIL-SERL = SAC (连续动作策略优化) 
         + DQN (离散夹爪Q学习) 
         + RLPD (混合数据采样)
         + 人类干预 (实时修正)
         + ResNet-10 (冻结视觉编码)
```

### 1.3 与标准 SAC 的主要差异
| 维度 | 标准 SAC | HIL-SERL |
|------|---------|----------|
| 动作空间 | 纯连续 | 混合 (连续+离散) |
| 网络架构 | Actor + Critic | Actor + Critic + Grasp Critic |
| 数据来源 | 纯在线 | 50% demo + 50% online |
| 视觉输入 | 可选 | ResNet-10 多相机 |
| 人类参与 | 无 | 实时干预 + 演示数据 |

---

## 2. 网络结构与数据流

### 2.1 输入数据规格

#### 观测空间
```
图像输入 (3个相机):
├─ wrist_1:     原始(1280, 720, 3) → 裁剪后可变
├─ wrist_2:     原始(1280, 720, 3) → 裁剪后可变
└─ side_policy: 原始(1280, 720, 3) → 裁剪后(250, 300, 3)

本体感知状态 (21维):
├─ tcp_pose (6):    末端位置(3) + 姿态欧拉角(3)
├─ tcp_vel (6):     线速度(3) + 角速度(3)
├─ tcp_force (3):   末端受力
├─ tcp_torque (3):  末端力矩
└─ gripper_pose (3): 左夹爪 + 右夹爪 + 占位符
```

#### 动作空间
```
连续动作 (12维):
├─ left_arm_ee (6):  左臂末端执行器增量 (位置3 + 姿态3)
└─ right_arm_ee (6): 右臂末端执行器增量

离散夹爪动作 (2维):
├─ left_gripper (1):  {-1=张开, 0=保持, 1=闭合}
└─ right_gripper (1): {-1=张开, 0=保持, 1=闭合}

环境实际动作 (14维):
[left_ee(6), left_gripper(1), right_ee(6), right_gripper(1)]
```

#### 夹爪联合动作编码
```
联合动作空间: 3 × 3 = 9 个状态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
joint_id | left_gripper | right_gripper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   0     |     -1       |     -1        (都张开)
   1     |     -1       |      0        (左张开, 右保持)
   2     |     -1       |      1        (左张开, 右闭合)
   3     |      0       |     -1        (左保持, 右张开)
   4     |      0       |      0        (都保持)
   5     |      0       |      1        (左保持, 右闭合)
   6     |      1       |     -1        (左闭合, 右张开)
   7     |      1       |      0        (左闭合, 右保持)
   8     |      1       |      1        (都闭合)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

编码公式:
joint_action = left_gripper * 3 + right_gripper
其中 left_gripper, right_gripper ∈ {0, 1, 2}
```

---

### 2.2 完整数据流：从原始输入到最终编码

#### 阶段 1: 图像预处理 (每个相机独立)
```
输入: 原始图像 (H_raw, W_raw, 3)
  ↓
相机特定裁剪: 任务定义的 ROI
  ↓ 
Resize 到标准尺寸: (128, 128, 3)
  ↓
ImageNet 标准化:
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  normalized = (pixel / 255.0 - mean) / std
  ↓
输出: (B, 128, 128, 3)
```

#### 阶段 2: ResNet-10 Frozen Encoder (冻结部分)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: (B, 128, 128, 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial Block:
├─ Conv2D(64, kernel=7×7, stride=2, padding=3):
│   输出: (B, 64, 64, 64)
│   参数: 7×7×3×64 + 64 = 9,472
├─ GroupNorm(num_groups=4):
│   输出: (B, 64, 64, 64)
│   参数: 2×64 = 128
├─ ReLU:
│   输出: (B, 64, 64, 64)
└─ MaxPool(kernel=3×3, stride=2, padding=1):
    输出: (B, 32, 32, 64)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 1: (1× ResNetBlock, filters=64, stride=1)
  ResNetBlock:
    ├─ Conv 3×3:        (B, 32, 32, 64)   # 3×3×64×64 = 36,864
    ├─ GroupNorm + ReLU
    ├─ Conv 3×3:        (B, 32, 32, 64)   # 3×3×64×64 = 36,864
    ├─ GroupNorm
    └─ Residual + ReLU: (B, 32, 32, 64)
  
  输出: (B, 32, 32, 64)
  参数小计: ~75K

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 2: (1× ResNetBlock, filters=128, stride=2)
  ResNetBlock:
    ├─ Conv 3×3, stride=2: (B, 16, 16, 128)  # 下采样
    ├─ GroupNorm + ReLU
    ├─ Conv 3×3:           (B, 16, 16, 128)
    ├─ GroupNorm
    ├─ Shortcut (匹配维度):
    │   ├─ Conv 1×1, stride=2: (B, 16, 16, 128)
    │   └─ GroupNorm
    └─ Residual + ReLU:    (B, 16, 16, 128)
  
  输出: (B, 16, 16, 128)
  参数小计: ~295K

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 3: (1× ResNetBlock, filters=256, stride=2)
  输出: (B, 8, 8, 256)
  参数小计: ~1.2M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stage 4: (1× ResNetBlock, filters=512, stride=2)
  输出: (B, 4, 4, 512)
  参数小计: ~3.4M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❄️  jax.lax.stop_gradient()  ← 梯度截断边界
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
冻结参数总计: ~5M (从 ImageNet 预训练加载)
```

#### 阶段 3: Spatial Learned Embeddings (可训练)
```
输入: (B, 4, 4, 512) ← 来自冻结的 ResNet-10

可学习 Kernel 参数:
  shape: (4, 4, 512, 8)  # 8 个空间特征块
  init: Lecun Normal
  参数量: 4 × 4 × 512 × 8 = 65,536

前向计算:
  1. features_expanded = expand_dims(features, axis=-1)
     shape: (B, 4, 4, 512, 1)
  
  2. kernel_expanded = expand_dims(kernel, axis=0)
     shape: (1, 4, 4, 512, 8)
  
  3. weighted = features_expanded × kernel_expanded
     shape: (B, 4, 4, 512, 8)
     # 广播乘法：每个空间位置的每个通道与8个权重相乘
  
  4. spatial_sum = sum(weighted, axis=(1, 2))
     shape: (B, 512, 8)
     # 对空间维度求和，保留通道和特征维度
  
  5. flatten = reshape(spatial_sum, [B, 512 × 8])
     shape: (B, 4096)
  
  6. dropout = Dropout(0.1, deterministic=not train)
     shape: (B, 4096)

输出: (B, 4096)
可训练参数: 65,536
```

#### 阶段 4: Bottleneck 降维 (可训练)
```
输入: (B, 4096)
  ↓
Dense(256):
  W: (4096, 256), b: (256)
  参数量: 4096 × 256 + 256 = 1,048,832
  输出: (B, 256)
  ↓
LayerNorm:
  γ: (256,), β: (256,)
  参数量: 512
  输出: (B, 256)
  ↓
Tanh 激活:
  输出: (B, 256)

输出: (B, 256) ← 单个相机的最终视觉编码
可训练参数: 1,049,344
```

#### 阶段 5: 多相机特征融合
```
wrist_1 编码:     (B, 256)
wrist_2 编码:     (B, 256)
side_policy 编码: (B, 256)
  ↓
Concatenate(axis=-1):
  ↓
视觉特征向量: (B, 768)
```

#### 阶段 6: 本体感知编码 (可训练)
```
输入: state (B, 21)
  ↓
Dense(64):
  init: Xavier Uniform
  参数量: 21 × 64 + 64 = 1,408
  输出: (B, 64)
  ↓
LayerNorm:
  参数量: 128 (γ + β)
  输出: (B, 64)
  ↓
Tanh 激活:
  输出: (B, 64)

输出: (B, 64) ← 本体感知编码
可训练参数: 1,536
```

#### 阶段 7: 最终观测编码
```
视觉特征:  (B, 768)
本体特征:  (B, 64)
  ↓
Concatenate(axis=-1):
  ↓
最终编码: (B, 832) ← obs_enc

这是所有下游网络(Actor/Critic/Grasp Critic)的共享输入
```

---

### 2.3 网络架构详解

#### 网络 1: Actor (策略网络)

**作用**: 输出连续动作的概率分布

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: obs_enc (B, 832)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLP Backbone:
├─ Layer 1:
│   ├─ Dense(256):  (B, 256)
│   │   参数: 832 × 256 + 256 = 213,248
│   └─ ReLU:        (B, 256)
│
└─ Layer 2:
    ├─ Dense(256):  (B, 256)
    │   参数: 256 × 256 + 256 = 65,792
    └─ ReLU:        (B, 256)  # activate_final=True

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Policy Head:
├─ 均值头:
│   └─ Dense(12):   (B, 12)
│       参数: 256 × 12 + 12 = 3,084
│       输出: μ(s) ∈ ℝ^12
│
└─ 标准差头 (std_parameterization="uniform"):
    └─ log_stds:    (12,)  # 全局可学习参数，不依赖状态
        输出: log σ ∈ ℝ^12
        σ = exp(log_stds)
        σ = clip(σ, 1e-5, 10.0) × √temperature

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

分布构建:
  base_dist = MultivariateNormalDiag(loc=μ, scale_diag=σ)
  policy_dist = TanhMultivariateNormalDiag(base_dist)
  # Tanh squash 将无界高斯映射到 [-1, 1]

采样:
  z ~ N(μ, σ)              # 从高斯采样
  action = tanh(z)         # Squash 到 [-1, 1]
  log_prob = log p(z) - log|det(∂tanh/∂z)|  # 变量变换修正

输出: 
  - 连续动作: (B, 12) ∈ [-1, 1]^12
  - 对数概率: (B,) 用于策略优化

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总参数: 213,248 + 65,792 + 3,084 + 12 = 282,136
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 网络 2: Critic (连续动作 Q 网络集成)

**作用**: 估计状态-动作对的 Q 值

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入:
  obs_enc: (B, 832)
  actions: (B, 12)  ← 仅连续动作部分，不含夹爪
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

拼接输入:
  concat([obs_enc, actions], axis=-1)
  输出: (B, 844)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ensemble 结构 (N=2 个独立 Critic):
  使用 vmap 并行计算，共享架构但参数独立
  
  单个 Critic:
  ├─ Layer 1:
  │   ├─ Dense(256):  (B, 256)
  │   │   参数: 844 × 256 + 256 = 216,320
  │   └─ ReLU:        (B, 256)
  │
  ├─ Layer 2:
  │   ├─ Dense(256):  (B, 256)
  │   │   参数: 256 × 256 + 256 = 65,792
  │   └─ ReLU:        (B, 256)
  │
  └─ Output:
      ├─ Dense(1):    (B, 1)
      │   参数: 256 × 1 + 1 = 257
      └─ Squeeze:     (B,)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ensemble 输出:
  Q_values: (2, B)  # 2个独立的Q估计
  
训练时使用:
  Q_min = min(Q_1, Q_2)  # Clipped Double Q-Learning
  用于计算 TD 目标，减少过估计偏差

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
单个 Critic 参数: 216,320 + 65,792 + 257 = 282,369
总参数 (×2): 564,738
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 网络 3: Grasp Critic (离散夹爪 Q 网络)

**作用**: 估计 9 个联合夹爪动作的 Q 值

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: obs_enc (B, 832)  ← 注意：不需要动作输入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLP Backbone (更小的网络):
├─ Layer 1:
│   ├─ Dense(128):  (B, 128)
│   │   参数: 832 × 128 + 128 = 106,624
│   └─ ReLU:        (B, 128)
│
└─ Layer 2:
    ├─ Dense(128):  (B, 128)
    │   参数: 128 × 128 + 128 = 16,512
    └─ ReLU:        (B, 128)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output Head:
  └─ Dense(9):      (B, 9)
      参数: 128 × 9 + 9 = 1,161
      输出: Q(s, a_joint) for a_joint ∈ {0, 1, ..., 8}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

动作选择 (推理时):
  Q_values = GraspCritic(obs)          # (B, 9)
  joint_action = argmax(Q_values)      # (B,) ∈ {0, ..., 8}
  
  # 解码为独立夹爪动作
  left_gripper = joint_action // 3 - 1   # (B,) ∈ {-1, 0, 1}
  right_gripper = joint_action % 3 - 1   # (B,) ∈ {-1, 0, 1}

训练时 (Double DQN):
  # 在线网络选动作
  next_Q_online = GraspCritic(next_obs)
  best_action = argmax(next_Q_online, axis=-1)
  
  # 目标网络评估
  next_Q_target = GraspCritic_target(next_obs)
  target_Q = next_Q_target[arange(B), best_action]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总参数: 106,624 + 16,512 + 1,161 = 124,297
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 网络 4: Temperature (熵系数)

**作用**: 自动调节探索-利用权衡

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Lagrange 乘子 (GeqLagrangeMultiplier):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

可学习参数:
  log_alpha: (1,)  # 对数空间确保 α > 0
  init_value: 1.0
  
前向传播:
  α = exp(log_alpha)  # 温度系数
  
目标:
  保持策略熵 H[π] ≥ target_entropy
  target_entropy = -action_dim / 2 = -6.0

损失函数:
  L_temp = α × (H[π] - target_entropy)
  # 如果熵太低 → α 增大 → 增加探索
  # 如果熵太高 → α 减小 → 增加利用

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总参数: 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 2.4 参数统计汇总

#### 按模块分类
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块                          状态      参数量        说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ResNet-10 Backbone (×3相机)   冻结❄️    ~5M × 3    ImageNet预训练
Spatial Embeddings (×3相机)   可训练🔥   65K × 3     可学习池化
Bottleneck (×3相机)           可训练🔥   1.05M × 3   降维投影
Proprio Encoder               可训练🔥   1.5K        状态编码
Actor MLP                     可训练🔥   282K        策略网络
Critic Ensemble (×2)          可训练🔥   565K        Q网络
Grasp Critic                  可训练🔥   124K        夹爪Q网络
Temperature                   可训练🔥   1           熵系数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
冻结参数总计:                           ~15M
可训练参数总计:                         ~4.4M
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

#### 编码器参数分布 (单相机)
```
ResNet-10 Frozen:     ~5,000,000  (冻结)
Spatial Embeddings:       65,536  (可训练)
Bottleneck:            1,049,344  (可训练)
────────────────────────────────
单相机总计:           ~6,114,880
可训练部分:            1,114,880
```

---

## 3. 训练流程

### 3.1 系统架构：分布式 Actor-Learner

```
┌─────────────────────────────────────────────────────────┐
│                     训练系统架构                          │
└─────────────────────────────────────────────────────────┘

┌──────────────────┐                    ┌──────────────────┐
│   Actor 进程     │                    │  Learner 进程     │
│  (环境交互)       │◄───────────────────│  (梯度更新)       │
│                  │   1. 初始网络参数    │                  │
│  ┌────────────┐  │   2. 定期同步参数    │  ┌────────────┐  │
│  │ Agent      │  │                    │  │ Agent      │  │
│  │ (推理)     │  │                    │  │ (训练)     │  │
│  └────────────┘  │                    │  └────────────┘  │
│        ↓         │                    │        ↑         │
│  ┌────────────┐  │                    │  ┌────────────┐  │
│  │ 环境       │  │                    │  │ 优化器     │  │
│  │ + 人类干预 │  │                    │  │ (Adam 3e-4)│  │
│  └────────────┘  │                    │  └────────────┘  │
│        ↓         │                    │        ↑         │
│  ┌────────────┐  │   3. 经验数据      │  ┌────────────┐  │
│  │ 数据队列   │──┼───────────────────►│  │ 回放缓冲区  │  │
│  │ (50K)      │  │                    │  │ (100K)     │  │
│  └────────────┘  │                    │  └────────────┘  │
│                  │                    │        ↑         │
│  ┌────────────┐  │   4. 演示数据      │  ┌────────────┐  │
│  │ 人类干预   │──┼───────────────────►│  │ 演示缓冲区  │  │
│  │ 数据队列    │  │                    │  │ (100K)     │  │
│  └────────────┘  │                    │  └────────────┘  │
└──────────────────┘                    └──────────────────┘

通信机制: TrainerServer / TrainerClient (agentlace)
同步频率: 每 steps_per_update 步 (默认1000步)
```

---

### 3.2 Actor 端训练循环

#### 初始化阶段
```python
# 1. 创建环境
env = USBEnv(config)
env = SpacemouseIntervention(env)      # 人类干预
env = RelativeFrame(env)                # 相对坐标系
env = Quat2EulerWrapper(env)            # 四元数→欧拉角
env = SERLObsWrapper(env)               # 观测标准化
env = ChunkingWrapper(env, obs_horizon=1)  # 单帧
env = RewardClassifierWrapper(env)      # 学习的奖励函数
env = GripperPenaltyWrapper(env)        # 夹爪惩罚

# 2. 创建 Agent
agent = SACAgentHybridDualArm.create_pixels(...)
agent = load_resnet10_params(agent)     # 加载预训练权重

# 3. 连接 Learner
client = TrainerClient(ip=learner_ip)
client.recv_network_callback(update_params)  # 注册参数更新回调

# 4. 创建数据队列
data_store = QueuedDataStore(capacity=50000)
intvn_data_store = QueuedDataStore(capacity=50000)
```

#### 主循环 (每步)
```python
for step in range(max_steps):
    # ─────────────────────────────────────────
    # 1. 动作采样
    # ─────────────────────────────────────────
    if step < random_steps:
        # 初期随机探索
        action = env.action_space.sample()
    else:
        # 策略采样
        rng, key = jax.random.split(rng)
        
        # 连续动作
        continuous_actions = agent.sample_actions(
            observations=obs,
            seed=key,
            argmax=False  # 探索模式
        )  # (12,) ∈ [-1, 1]^12
        
        # 夹爪动作 (内部已集成)
        # action 已包含 14 维: [left_ee(6), left_grip(1), right_ee(6), right_grip(1)]
        action = continuous_actions
    
    # ─────────────────────────────────────────
    # 2. 环境交互
    # ─────────────────────────────────────────
    next_obs, reward, done, truncated, info = env.step(action)
    
    # ─────────────────────────────────────────
    # 3. 人类干预检测
    # ─────────────────────────────────────────
    if "intervene_action" in info:
        # 人类通过 SpaceMouse 修正了动作
        action = info.pop("intervene_action")
        intervention_count += 1
        is_intervention = True
    else:
        is_intervention = False
    
    # ─────────────────────────────────────────
    # 4. 构造 Transition
    # ─────────────────────────────────────────
    transition = {
        "observations": obs,
        "actions": action,              # (14,)
        "next_observations": next_obs,
        "rewards": reward,
        "masks": 1.0 - done,
        "dones": done,
    }
    
    # 添加夹爪惩罚 (如果有)
    if "grasp_penalty" in info:
        transition["grasp_penalty"] = info["grasp_penalty"]
    
    # ─────────────────────────────────────────
    # 5. 数据存储
    # ─────────────────────────────────────────
    data_store.insert(transition)           # 所有数据
    if is_intervention:
        intvn_data_store.insert(transition) # 人类干预数据
    
    # ─────────────────────────────────────────
    # 6. 参数同步 (被动接收)
    # ─────────────────────────────────────────
    # Learner 会定期推送新参数
    # update_params() 回调自动更新 agent.state.params
    
    obs = next_obs
    
    if done or truncated:
        obs, _ = env.reset()
```

---

### 3.3 Learner 端训练循环

#### 初始化阶段
```python
# 1. 创建 Agent (同 Actor)
agent = SACAgentHybridDualArm.create_pixels(...)
agent = load_resnet10_params(agent)

# 2. 加载演示数据
demo_buffer = MemoryEfficientReplayBuffer(capacity=100000)
for demo_file in demo_paths:
    transitions = load(demo_file)
    for trans in transitions:
        demo_buffer.insert(trans)

# 3. 创建在线数据缓冲区
replay_buffer = MemoryEfficientReplayBuffer(capacity=100000)

# 4. 启动服务器
server = TrainerServer(config)
server.register_data_store("actor_env", replay_buffer)
server.register_data_store("actor_env_intvn", demo_buffer)
server.start(threaded=True)

# 5. 等待初始数据
while len(replay_buffer) < training_starts:
    time.sleep(1)

# 6. 发送初始网络
server.publish_network(agent.state.params)
```

#### 主循环 (每步)
```python
for step in range(max_steps):
    # ═════════════════════════════════════════════════════════
    # 阶段 1: 更新 Critic (cta_ratio - 1 次)
    # ═════════════════════════════════════════════════════════
    for _ in range(cta_ratio - 1):  # 默认 cta_ratio=2, 执行1次
        # ─────────────────────────────────────────
        # 1.1 采样数据 (RLPD 50/50 混合)
        # ─────────────────────────────────────────
        online_batch = replay_buffer.sample(batch_size // 2)
        demo_batch = demo_buffer.sample(batch_size // 2)
        batch = concat_batches(online_batch, demo_batch)
        # batch_size = 256 (128 online + 128 demo)
        
        # ─────────────────────────────────────────
        # 1.2 更新 Critic 和 Grasp Critic
        # ─────────────────────────────────────────
        agent, critic_info = agent.update(
            batch,
            networks_to_update=frozenset({"critic", "grasp_critic"})
        )
    
    # ═════════════════════════════════════════════════════════
    # 阶段 2: 更新所有网络 (1 次)
    # ═════════════════════════════════════════════════════════
    online_batch = replay_buffer.sample(batch_size // 2)
    demo_batch = demo_buffer.sample(batch_size // 2)
    batch = concat_batches(online_batch, demo_batch)
    
    agent, update_info = agent.update(
        batch,
        networks_to_update=frozenset({
            "critic", "grasp_critic", "actor", "temperature"
        })
    )
    
    # ═════════════════════════════════════════════════════════
    # 阶段 3: 同步参数到 Actor
    # ═════════════════════════════════════════════════════════
    if step % steps_per_update == 0:  # 默认 1000 步
        agent = jax.block_until_ready(agent)  # 确保计算完成
        server.publish_network(agent.state.params)
    
    # ═════════════════════════════════════════════════════════
    # 阶段 4: 日志记录
    # ═════════════════════════════════════════════════════════
    if step % log_period == 0:
        wandb_logger.log(update_info, step=step)
    
    # ═════════════════════════════════════════════════════════
    # 阶段 5: 保存检查点
    # ═════════════════════════════════════════════════════════
    if step % checkpoint_period == 0:  # 默认 2000 步
        save_checkpoint(checkpoint_path, agent.state, step=step)
```

---

## 4. 参数更新机制

### 4.1 单次更新的完整流程

#### 数据准备
```python
# 输入: batch (混合采样)
batch = {
    "observations": {
        "wrist_1": (256, 128, 128, 3),
        "wrist_2": (256, 128, 128, 3),
        "side_policy": (256, 128, 128, 3),
        "state": (256, 21),
    },
    "actions": (256, 14),           # [ee(12), gripper(2)]
    "next_observations": {...},     # 同上
    "rewards": (256,),
    "masks": (256,),                # 1 - done
    "grasp_penalty": (256,),        # 夹爪切换惩罚
}

# 数据增强 (可选)
if augmentation_function is not None:
    batch = augmentation_function(batch, rng)
    # 随机裁剪 + 颜色抖动

# 奖励偏置
batch["rewards"] = batch["rewards"] + reward_bias  # 默认 0.0
```

---

### 4.2 Critic 损失计算与更新

#### 前向传播
```python
# ─────────────────────────────────────────
# 1. 提取连续动作 (忽略夹爪维度)
# ─────────────────────────────────────────
actions_continuous = jnp.concatenate([
    batch["actions"][:, :6],    # left_ee
    batch["actions"][:, 7:13],  # right_ee
], axis=-1)  # (256, 12)

# ─────────────────────────────────────────
# 2. 计算 next_actions 和 log_probs
# ─────────────────────────────────────────
rng, key = jax.random.split(rng)
next_action_dist = agent.forward_policy(
    batch["next_observations"], rng=key
)
next_actions, next_log_probs = next_action_dist.sample_and_log_prob(seed=key)
# next_actions: (256, 12)
# next_log_probs: (256,)

# ─────────────────────────────────────────
# 3. 计算 Target Q (使用目标网络)
# ─────────────────────────────────────────
target_next_qs = agent.forward_target_critic(
    batch["next_observations"],
    next_actions,
    rng=key
)  # (2, 256) - 集成的两个 Critic

# 取最小值 (pessimistic estimate)
target_next_q = target_next_qs.min(axis=0)  # (256,)

# ─────────────────────────────────────────
# 4. 计算 TD Target
# ─────────────────────────────────────────
target_q = batch["rewards"] + discount * batch["masks"] * target_next_q
# target_q: (256,)

if backup_entropy:  # 默认 False
    temperature = agent.forward_temperature()
    target_q = target_q - temperature * next_log_probs

# ─────────────────────────────────────────
# 5. 计算当前 Q 预测
# ─────────────────────────────────────────
predicted_qs = agent.forward_critic(
    batch["observations"],
    actions_continuous,
    rng=key,
    grad_params=params  # 梯度计算使用的参数
)  # (2, 256)

# ─────────────────────────────────────────
# 6. 计算 MSE 损失
# ─────────────────────────────────────────
target_qs = target_q[None].repeat(2, axis=0)  # (2, 256)
critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
```

#### 梯度更新
```python
# 计算梯度
grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
(loss, info), grads = grad_fn(agent.state.params["modules_critic"])

# Adam 更新
updates, opt_state = optimizer.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)

# 软更新目标网络
target_params = (
    soft_target_update_rate * new_params +
    (1 - soft_target_update_rate) * target_params
)
# soft_target_update_rate = 0.005 (τ)
```

---

### 4.3 Grasp Critic 损失计算与更新

#### 前向传播
```python
# ─────────────────────────────────────────
# 1. 解析夹爪动作
# ─────────────────────────────────────────
# 环境动作 [-1, 0, 1] → DQN 动作 {0, 1, 2}
grasp_action1 = jnp.round(batch["actions"][:, 6]).astype(jnp.int16) + 1
grasp_action2 = jnp.round(batch["actions"][:, 13]).astype(jnp.int16) + 1

# 编码为联合动作
joint_grasp_action = grasp_action1 * 3 + grasp_action2  # (256,) ∈ {0, ..., 8}

# ─────────────────────────────────────────
# 2. Double DQN: 在线网络选动作
# ─────────────────────────────────────────
next_grasp_qs_online = agent.forward_grasp_critic(
    batch["next_observations"], rng=key
)  # (256, 9)

best_next_action = next_grasp_qs_online.argmax(axis=-1)  # (256,)

# ─────────────────────────────────────────
# 3. 目标网络评估
# ─────────────────────────────────────────
next_grasp_qs_target = agent.forward_target_grasp_critic(
    batch["next_observations"], rng=key
)  # (256, 9)

target_next_q = next_grasp_qs_target[
    jnp.arange(batch_size), best_next_action
]  # (256,)

# ─────────────────────────────────────────
# 4. 计算 TD Target (包含夹爪惩罚)
# ─────────────────────────────────────────
grasp_rewards = batch["rewards"] + batch["grasp_penalty"]
target_grasp_q = grasp_rewards + discount * batch["masks"] * target_next_q

# ─────────────────────────────────────────
# 5. 当前 Q 预测
# ─────────────────────────────────────────
predicted_grasp_qs = agent.forward_grasp_critic(
    batch["observations"], rng=key, grad_params=params
)  # (256, 9)

predicted_q = predicted_grasp_qs[
    jnp.arange(batch_size), joint_grasp_action
]  # (256,)

# ─────────────────────────────────────────
# 6. MSE 损失
# ─────────────────────────────────────────
grasp_critic_loss = jnp.mean((predicted_q - target_grasp_q) ** 2)
```

#### 梯度更新
```python
# 同 Critic，使用独立的优化器
grad_fn = jax.value_and_grad(grasp_critic_loss_fn, has_aux=True)
(loss, info), grads = grad_fn(agent.state.params["modules_grasp_critic"])

updates, opt_state = optimizer.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
```

---

### 4.4 Actor 损失计算与更新

#### 前向传播
```python
# ─────────────────────────────────────────
# 1. 获取温度系数
# ─────────────────────────────────────────
temperature = agent.forward_temperature()  # α

# ─────────────────────────────────────────
# 2. 策略采样
# ─────────────────────────────────────────
rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)

action_dist = agent.forward_policy(
    batch["observations"], 
    rng=policy_rng, 
    grad_params=params
)
actions, log_probs = action_dist.sample_and_log_prob(seed=sample_rng)
# actions: (256, 12)
# log_probs: (256,)

# ─────────────────────────────────────────
# 3. 评估 Q 值
# ─────────────────────────────────────────
predicted_qs = agent.forward_critic(
    batch["observations"],
    actions,
    rng=critic_rng
)  # (2, 256)

# 使用集成平均 (而非最小值)
predicted_q = predicted_qs.mean(axis=0)  # (256,)

# ─────────────────────────────────────────
# 4. SAC 目标
# ─────────────────────────────────────────
actor_objective = predicted_q - temperature * log_probs
actor_loss = -jnp.mean(actor_objective)  # 梯度上升 → 损失下降
```

#### 梯度更新
```python
grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
(loss, info), grads = grad_fn(agent.state.params["modules_actor"])

updates, opt_state = optimizer.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
```

---

### 4.5 Temperature 更新

#### 损失计算
```python
# ─────────────────────────────────────────
# 1. 计算当前策略熵
# ─────────────────────────────────────────
rng, key = jax.random.split(rng)
next_actions, next_log_probs = agent._compute_next_actions(batch, key)

entropy = -next_log_probs.mean()  # H[π]

# ─────────────────────────────────────────
# 2. Lagrange 惩罚
# ─────────────────────────────────────────
temperature_loss = agent.temperature_lagrange_penalty(
    lhs=entropy,
    rhs=target_entropy,  # -6.0
    grad_params=params
)
# temperature_loss = α × (entropy - target_entropy)

# 如果 entropy < target_entropy → loss < 0 → α 减小
# 如果 entropy > target_entropy → loss > 0 → α 增大
```

#### 梯度更新
```python
grad_fn = jax.value_and_grad(temperature_loss_fn, has_aux=True)
(loss, info), grads = grad_fn(agent.state.params["modules_temperature"])

updates, opt_state = optimizer.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)
```

---

### 4.6 更新顺序与频率

#### 每个训练步的更新模式
```
步骤 1 到 (cta_ratio - 1):
  ├─ 采样 batch
  ├─ 更新 Critic
  └─ 更新 Grasp Critic

步骤 cta_ratio:
  ├─ 采样 batch
  ├─ 更新 Critic
  ├─ 更新 Grasp Critic
  ├─ 更新 Actor
  └─ 更新 Temperature

目标网络软更新:
  每次更新 Critic 后执行
  θ_target ← τ × θ + (1 - τ) × θ_target
```

#### 更新频率配置
```python
cta_ratio = 2  # Critic-to-Actor 比例
# → 每轮训练: 2次 Critic 更新, 1次 Actor 更新

steps_per_update = 1000  # Actor 参数同步间隔
# → Learner 每训练 1000 步推送一次新参数

checkpoint_period = 2000  # 检查点保存间隔
log_period = 100  # 日志记录间隔
```

---

### 4.7 梯度流与冻结边界

```
┌─────────────────────────────────────────────────┐
│               前向传播路径                        │
└─────────────────────────────────────────────────┘

观测 → ResNet-10 → stop_gradient → Spatial Emb → Bottleneck → obs_enc
        (冻结❄️)      ↑截断边界        (可训练🔥)  (可训练🔥)
                                                      ↓
                                    ┌─────────────────┼─────────────────┐
                                    ↓                 ↓                 ↓
                                 Actor            Critic         Grasp Critic
                                (可训练🔥)        (可训练🔥)       (可训练🔥)

┌─────────────────────────────────────────────────┐
│               反向传播路径                        │
└─────────────────────────────────────────────────┘

Actor Loss → ∂L/∂θ_actor → ∂L/∂obs_enc → ∂L/∂bottleneck → ∂L/∂spatial_emb
                                                                   ↓
                                                            stop_gradient ✋
                                                            梯度不再向下传播

关键设计:
1. ResNet-10 完全冻结 → 保留预训练的通用视觉特征
2. Spatial Embeddings 可训练 → 学习任务特定的空间聚合
3. 所有下游网络共享编码器 → 梯度汇聚，加速学习
```

---

## 5. 关键超参数

### 5.1 网络架构参数
```python
# 编码器
encoder_type = "resnet-pretrained"
num_spatial_blocks = 8              # Spatial Embeddings 特征数
bottleneck_dim = 256                # 单相机编码维度
use_proprio = True                  # 使用本体感知
proprio_latent_dim = 64             # 本体感知编码维度

# Actor
actor_hidden_dims = [256, 256]
std_parameterization = "uniform"    # 全局可学习标准差
tanh_squash_distribution = True     # Tanh squash 到 [-1, 1]

# Critic
critic_hidden_dims = [256, 256]
critic_ensemble_size = 2            # 集成数量
critic_subsample_size = None        # 不使用子采样

# Grasp Critic
grasp_critic_hidden_dims = [128, 128]
output_dim = 9                      # 联合夹爪动作数
```

---

### 5.2 训练超参数
```python
# 优化器
learning_rate = 3e-4                # 所有网络统一学习率
optimizer = "adam"                  # Adam 优化器

# RL 算法
discount = 0.98                     # 折扣因子 γ
soft_target_update_rate = 0.005     # 目标网络更新率 τ
target_entropy = -6.0               # 目标熵 (-action_dim / 2)
backup_entropy = False              # 不在 TD target 中减去熵

# 训练流程
batch_size = 256                    # 训练批次大小
training_starts = 5000              # 开始训练前的初始数据量
random_steps = 0                    # 随机探索步数
cta_ratio = 2                       # Critic-to-Actor 更新比例
steps_per_update = 1000             # Actor 参数同步间隔
max_steps = 1_000_000               # 最大训练步数

# 数据管理
replay_buffer_capacity = 100000     # 在线数据容量
demo_buffer_capacity = 100000       # 演示数据容量
buffer_period = 1000                # 数据持久化间隔

# 日志与检查点
log_period = 100                    # 日志记录间隔
checkpoint_period = 2000            # 检查点保存间隔
```

---

### 5.3 环境与任务参数
```python
# 观测配置
image_keys = ["side_policy", "wrist_1", "wrist_2"]
classifier_keys = ["side_classifier"]
proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", 
                "tcp_torque", "gripper_pose"]
obs_horizon = 1                     # 单帧观测
act_exec_horizon = None             # 单步动作

# 奖励配置
reward_bias = 0.0                   # 奖励偏置
grasp_penalty = -0.02               # 夹爪切换惩罚

# 动作空间
action_scale = [0.015, 0.1, 1]      # [位置, 旋转, 夹爪]
max_episode_length = 120            # 最大回合长度
```

---

### 5.4 数据增强
```python
augmentation_function = batched_random_crop  # 随机裁剪
padding = 4                                  # 裁剪填充
# 从 (128, 128) 填充到 (136, 136) 后随机裁剪回 (128, 128)

# 可选: 颜色抖动
# brightness, contrast, saturation, hue jitter
```

---

### 5.5 RLPD 混合采样
```python
# 50/50 混合采样策略
online_batch_size = batch_size // 2  # 128
demo_batch_size = batch_size // 2    # 128

# 数据来源
online_data:  在线交互收集 (Actor 采样)
demo_data:    人类演示 + 人类干预修正

# 混合比例固定 1:1
# 保证策略既能学习人类专家行为，又能通过探索发现新策略
```

---

## 6. 训练监控指标

### 6.1 Actor 端指标
```python
"environment": {
    "episode/return": float,              # 回合总奖励
    "episode/length": int,                # 回合长度
    "episode/intervention_count": int,    # 人类干预次数
    "episode/intervention_steps": int,    # 人类干预总步数
}

"timer": {
    "sample_actions": float,              # 动作采样耗时 (ms)
    "step_env": float,                    # 环境步进耗时 (ms)
    "total": float,                       # 总循环耗时 (ms)
}
```

### 6.2 Learner 端指标
```python
"critic_loss": float,                     # Critic MSE 损失
"predicted_qs": float,                    # 预测 Q 值均值
"target_qs": float,                       # 目标 Q 值均值
"rewards": float,                         # 批次奖励均值

"grasp_critic_loss": float,               # Grasp Critic MSE 损失
"predicted_grasp_qs": float,              # 预测夹爪 Q 值均值
"target_grasp_qs": float,                 # 目标夹爪 Q 值均值
"grasp_rewards": float,                   # 夹爪奖励均值 (含惩罚)

"actor_loss": float,                      # Actor 策略损失
"temperature": float,                     # 当前温度系数 α
"entropy": float,                         # 策略熵 H[π]

"temperature_loss": float,                # 温度损失

"actor_lr": float,                        # Actor 学习率
"critic_lr": float,                       # Critic 学习率
"grasp_critic_lr": float,                 # Grasp Critic 学习率
"temperature_lr": float,                  # Temperature 学习率

"timer": {
    "sample_replay_buffer": float,        # 数据采样耗时
    "train_critics": float,               # Critic 训练耗时
    "train": float,                       # 完整更新耗时
}
```

---

## 7. 关键设计决策

### 7.1 为什么冻结 ResNet-10？
1. **预训练特征足够** → ImageNet 学到的边缘/纹理/形状对机器人任务有效
2. **防止过拟合** → 机器人数据集小，微调大模型容易过拟合
3. **训练稳定** → 减少可训练参数，加快收敛
4. **计算效率** → 冻结层不计算梯度，节省内存和时间

### 7.2 为什么用 Spatial Learned Embeddings？
1. **保留空间信息** → 普通池化丢失太多信息 (4×4×512 → 512)
2. **可学习适应** → 学习任务特定的空间聚合方式
3. **参数效率** → 65K 参数即可，比全连接层小得多

### 7.3 为什么混合动作空间？
1. **连续动作 (SAC)** → 适合精细的末端执行器控制
2. **离散夹爪 (DQN)** → 夹爪只有3个状态，离散建模更高效
3. **联合编码** → 3×3=9 避免组合爆炸，同时捕捉双臂协同

### 7.4 为什么 RLPD 50/50 混合？
1. **演示数据** → 提供良好的初始策略，加速训练
2. **在线数据** → 允许探索和发现比人类演示更好的策略
3. **平衡** → 50/50 防止过度依赖演示或过度探索

### 7.5 为什么单帧观测？
1. **本体感知包含速度** → 不需要多帧差分
2. **实时反馈** → 每步都能修正，不需要长时序
3. **训练效率** → 减少内存和计算
4. **预训练兼容** → ResNet 第一层保持 3 通道

---

## 8. 与相关算法对比

### 8.1 vs 标准 SAC
| 维度 | 标准 SAC | HIL-SERL |
|------|---------|----------|
| 动作空间 | 纯连续 | 混合 (连续+离散) |
| 数据来源 | 纯在线 | 50% demo + 50% online |
| 视觉输入 | 可选/简单 | 多相机 ResNet-10 |
| 人类参与 | 无 | SpaceMouse 实时干预 |

### 8.2 vs 纯模仿学习 (BC)
| 维度 | BC | HIL-SERL |
|------|-----|----------|
| 学习方式 | 监督学习 | 强化学习 + 演示 |
| 性能上限 | 受限于演示 | 可超越人类演示 |
| 数据效率 | 高 (仅需演示) | 中 (需在线交互) |
| 泛化能力 | 弱 | 强 (通过探索) |

### 8.3 vs ACT / Diffusion Policy
| 维度 | ACT/Diffusion | HIL-SERL |
|------|---------------|----------|
| 动作输出 | 序列 (10步) | 单步 |
| 观测输入 | 多帧 (3帧) | 单帧 |
| 反馈模式 | 开环执行 | 闭环实时 |
| 适用任务 | 轨迹跟踪 | 精细操作 |

---

## 附录

### A. 完整数据维度流
```
原始输入:
├─ wrist_1:     (B, 1280, 720, 3)
├─ wrist_2:     (B, 1280, 720, 3)
├─ side_policy: (B, 1280, 720, 3)
└─ state:       (B, 21)

↓ 预处理 + ResNet-10

每个图像 → (B, 4, 4, 512) → (B, 4096) → (B, 256)

↓ 融合

视觉: (B, 768), 本体: (B, 64)

↓ 最终编码

obs_enc: (B, 832)

↓ 下游网络

Actor:        (B, 832) → (B, 12)
Critic:       (B, 832+12) → (2, B)
Grasp Critic: (B, 832) → (B, 9)
```

### B. 参数总量
```
冻结:     ~15M (ResNet-10 × 3 cameras × 3 networks)
可训练:   ~4.4M
总计:     ~19.4M
```

---

**文档版本**: v1.0  
**最后更新**: 2025-01  
**适用代码版本**: hil-serl main branch
