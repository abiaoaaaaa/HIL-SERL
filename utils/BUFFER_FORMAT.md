# HIL-SERL Replay Buffer 数据格式详解

## 概述

HIL-SERL 使用 pickle 格式存储 replay buffer 和 demo buffer，用于离线训练和策略学习。本文档详细说明数据的存储结构和格式。

## 文件组织

```
checkpoints/
├── buffer/                          # 主 buffer（Policy + 人工演示）
│   ├── transitions_1000.pkl        # 每 1000 步保存一次
│   ├── transitions_2000.pkl
│   └── ...
└── demo_buffer/                     # 仅人工演示数据
    ├── transitions_1000.pkl
    ├── transitions_2000.pkl
    └── ...
```

### 文件命名规则
- 格式：`transitions_<step>.pkl`
- `<step>` 是训练步数（learner step），不是环境步数
- 每个文件约 376 MB（包含 ~1000 个 transitions）

## 数据结构

### 顶层结构

每个 `.pkl` 文件包含一个 **Python list**，其中每个元素是一个 **transition (dict)**。

```python
# 文件内容
transitions = [
    transition_0,  # dict
    transition_1,  # dict
    transition_2,  # dict
    ...
]

# 典型大小
len(transitions) ≈ 1000~1001
```

### Transition 结构

每个 transition 是一个字典，包含以下键：

```python
transition = {
    'observations': dict,           # 当前时刻的观测
    'actions': np.ndarray,          # 执行的动作
    'next_observations': dict,      # 下一时刻的观测
    'rewards': int/float,           # 奖励值
    'dones': int (0 or 1),         # 是否终止
    'masks': float,                 # 掩码（通常与 dones 相关）
    'grasp_penalty': float,         # 夹爪惩罚项（任务相关）
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `observations` | dict | 当前状态的多模态观测（图像 + state） |
| `actions` | np.ndarray | 7维动作向量 [dx, dy, dz, drx, dry, drz, gripper] |
| `next_observations` | dict | 执行动作后的观测 |
| `rewards` | int/float | 奖励信号（0 或 1，由分类器判断） |
| `dones` | int | Episode 是否结束（0=继续, 1=终止） |
| `masks` | float | 掩码值（通常 1-dones） |
| `grasp_penalty` | float | 额外的惩罚项 |

---

## Observations 结构

`observations` 和 `next_observations` 是字典，包含多个相机图像和机器人状态：

```python
observations = {
    'wrist_1': np.ndarray,          # 手腕相机 1
    'wrist_2': np.ndarray,          # 手腕相机 2
    'side_policy': np.ndarray,      # 侧视相机（用于 policy）
    'side_classifier': np.ndarray,  # 侧视相机（用于 classifier）
    'state': np.ndarray,            # 机器人状态向量
}
```

### 图像数据

所有图像都是 **uint8 格式**，已经过裁剪和缩放处理：

| 字段 | Shape | 说明 |
|------|-------|------|
| `wrist_1` | `(1, 128, 128, 3)` | 手腕相机 1 的 RGB 图像 |
| `wrist_2` | `(1, 128, 128, 3)` | 手腕相机 2 的 RGB 图像 |
| `side_policy` | `(1, 128, 128, 3)` | 侧视相机图像（policy 输入） |
| `side_classifier` | `(1, 128, 128, 3)` | 侧视相机图像（classifier 输入） |

**注意**：
- 第一维是 batch 维度（训练时使用）
- 图像已归一化到 [0, 255] 的 uint8 格式
- 使用时需要转换为 float 并归一化到 [0, 1] 或 [-1, 1]
- 不同相机可能使用不同的裁剪函数

### State 数据

`state` 是一个 **19 维的 float32 向量**，包含机器人的低维状态：

```python
state.shape = (1, 19)
state.dtype = float32
```

#### State 维度解析（19 维）

| 索引 | 维度 | 名称 | 说明 | 单位 |
|------|------|------|------|------|
| 0 | 1 | `gripper` | 夹爪开合状态 | [0, 1] (0=关闭, 1=打开) |
| 1-3 | 3 | `force` | 末端执行器受力 | N (牛顿) |
| 4-9 | 6 | `tcp_pose` | 末端位姿 (x, y, z, rx, ry, rz) | m, rad |
| 10-12 | 3 | `torque` | 关节力矩 | Nm |
| 13-18 | 6 | `tcp_vel` | 末端速度 (vx, vy, vy, wx, wy, wz) | m/s, rad/s |

**备注**：
- `tcp_pose` 的前 3 维是笛卡尔位置（基座坐标系）
- `tcp_pose` 的后 3 维是欧拉角姿态（ZYX 顺序）
- 速度是在基座坐标系下的线速度和角速度

#### 替代 State 格式（13 维）

某些配置可能使用简化的 13 维 state：

| 索引 | 维度 | 名称 | 说明 |
|------|------|------|------|
| 0 | 1 | `gripper` | 夹爪状态 |
| 1-6 | 6 | `tcp_pose` | 末端位姿 |
| 7-12 | 6 | `tcp_vel` | 末端速度 |

---

## Actions 结构

`actions` 是一个 **7 维的 float32 向量**：

```python
actions.shape = (7,)
actions.dtype = float32
```

### 动作维度

| 索引 | 名称 | 说明 | 范围 |
|------|------|------|------|
| 0 | `dx` | X 方向增量（前后） | [-1, 1] |
| 1 | `dy` | Y 方向增量（左右） | [-1, 1] |
| 2 | `dz` | Z 方向增量（上下） | [-1, 1] |
| 3 | `drx` | 绕 X 轴旋转增量 (Roll) | [-1, 1] |
| 4 | `dry` | 绕 Y 轴旋转增量 (Pitch) | [-1, 1] |
| 5 | `drz` | 绕 Z 轴旋转增量 (Yaw) | [-1, 1] |
| 6 | `gripper` | 夹爪命令 | [-1, 1] (-1=关闭, 1=打开) |

**坐标系**：
- 增量在**末端执行器坐标系**下定义
- 实际执行时会转换到基座坐标系
- 数值已归一化到 [-1, 1]，实际物理增量由环境缩放

**示例值**：
```python
actions = [-0.84, -0.22, 0.02, -0.83, -0.89, -0.97, -0.33]
#          ^^^^^ 平移增量 ^^^^^  ^^^^^^ 旋转增量 ^^^^^^  ^^^ 夹爪
```

---

## Rewards 结构

### 奖励类型

```python
rewards: int or float
```

- **0**: 失败或中间状态
- **1**: 成功（通常由分类器判断）

### 奖励来源

HIL-SERL 使用 **learned reward classifier** 判断任务成功：
1. 从 `side_classifier` 图像提取特征
2. 输入奖励分类器网络
3. 输出 logit 值，经过 sigmoid 转换为概率
4. 如果 sigmoid > 阈值（如 0.5），则 reward=1

---

## 文件大小和内存占用

### 单个 Transition 的内存占用

```
图像数据:
  - wrist_1:          128×128×3 × 1 byte  = 48 KB
  - wrist_2:          128×128×3 × 1 byte  = 48 KB
  - side_policy:      128×128×3 × 1 byte  = 48 KB
  - side_classifier:  128×128×3 × 1 byte  = 48 KB
  小计: 192 KB

State 数据:
  - observations.state:      19 × 4 bytes = 76 bytes
  - next_observations.state: 19 × 4 bytes = 76 bytes
  小计: 152 bytes

Actions:
  - actions: 7 × 4 bytes = 28 bytes

其他标量:
  - rewards, dones, masks, grasp_penalty: ~20 bytes

总计: ~192 KB (主要是图像)
```

### 文件级统计

- **每个文件**: ~1000 transitions
- **文件大小**: ~376 MB
- **总 buffer 容量**: 通常 20-60 个文件（2万-6万 transitions）

---

## 数据加载示例

### 基本加载

```python
import pickle
import numpy as np

# 加载单个文件
with open('transitions_1000.pkl', 'rb') as f:
    transitions = pickle.load(f)

print(f"Loaded {len(transitions)} transitions")

# 访问第一个 transition
t = transitions[0]
print("Keys:", t.keys())
print("Action:", t['actions'])
print("Reward:", t['rewards'])
print("Done:", t['dones'])

# 访问观测
obs = t['observations']
print("State:", obs['state'])
print("Image keys:", obs.keys())
```

### 批量加载和拼接

```python
import glob
import pickle
import numpy as np

# 加载所有文件
buffer_dir = "checkpoints/buffer/"
files = sorted(glob.glob(f"{buffer_dir}/transitions_*.pkl"))

all_actions = []
all_states = []
all_rewards = []

for file in files:
    with open(file, 'rb') as f:
        transitions = pickle.load(f)
    
    # 提取数据
    for t in transitions:
        all_actions.append(t['actions'])
        all_states.append(t['observations']['state'].flatten())
        all_rewards.append(t['rewards'])

# 转换为 numpy 数组
actions = np.stack(all_actions)      # (N, 7)
states = np.stack(all_states)        # (N, 19)
rewards = np.array(all_rewards)      # (N,)

print(f"Total transitions: {len(actions)}")
print(f"Success rate: {rewards.mean()*100:.1f}%")
```

### 区分 Policy 和人工演示

```python
import hashlib

def transition_signature(t):
    """计算 transition 的唯一签名"""
    digest = hashlib.blake2b(digest_size=16)
    
    # 使用 state + action + next_state 计算哈希
    state = t['observations']['state'].flatten()
    action = t['actions'].flatten()
    next_state = t['next_observations']['state'].flatten()
    
    for arr in [state, action, next_state]:
        arr = np.ascontiguousarray(arr)
        digest.update(str(arr.dtype).encode('ascii'))
        digest.update(str(arr.shape).encode('ascii'))
        digest.update(arr.tobytes())
    
    return digest.digest()

# 加载主 buffer 和 demo buffer
with open('buffer/transitions_1000.pkl', 'rb') as f:
    main_buffer = pickle.load(f)

with open('demo_buffer/transitions_1000.pkl', 'rb') as f:
    demo_buffer = pickle.load(f)

# 构建 demo 签名集合
demo_signatures = {transition_signature(t) for t in demo_buffer}

# 标记每个 transition 是否为人工演示
is_human = np.array([
    transition_signature(t) in demo_signatures 
    for t in main_buffer
])

print(f"Policy steps: {(~is_human).sum()}")
print(f"Human steps: {is_human.sum()}")
print(f"Human ratio: {is_human.mean()*100:.1f}%")
```

---

## Episode 切分

### 使用 dones 标记

```python
def split_episodes_by_dones(transitions):
    """按 dones 切分 episodes"""
    episodes = []
    current_episode = []
    
    for t in transitions:
        current_episode.append(t)
        if t['dones'] == 1:
            episodes.append(current_episode)
            current_episode = []
    
    # 处理未完成的 episode
    if current_episode:
        episodes.append(current_episode)
    
    return episodes

episodes = split_episodes_by_dones(transitions)
print(f"Found {len(episodes)} episodes")
print(f"Episode lengths: {[len(ep) for ep in episodes[:5]]}")
```

### 使用观测连续性

```python
def is_continuous(t1, t2, atol=1e-6):
    """检查两个 transition 是否连续"""
    # 检查 t1 的 next_state 是否等于 t2 的 state
    next_state = t1['next_observations']['state'].flatten()
    curr_state = t2['observations']['state'].flatten()
    
    return np.allclose(next_state, curr_state, atol=atol, rtol=1e-5)

def split_episodes_with_continuity(transitions):
    """同时使用 dones 和连续性切分"""
    episodes = []
    current_episode = [transitions[0]]
    
    for i in range(1, len(transitions)):
        t_prev = transitions[i-1]
        t_curr = transitions[i]
        
        # 检查是否应该开始新 episode
        if t_prev['dones'] == 1 or not is_continuous(t_prev, t_curr):
            episodes.append(current_episode)
            current_episode = [t_curr]
        else:
            current_episode.append(t_curr)
    
    if current_episode:
        episodes.append(current_episode)
    
    return episodes
```

---

## 常见问题

### Q1: 为什么图像的第一维是 1？

**A**: 这是为了与训练时的 batch 维度保持一致。训练时会将多个样本拼接成 batch，例如：
```python
# 单个样本
image.shape = (1, 128, 128, 3)

# Batch (N=32)
batch_images.shape = (32, 128, 128, 3)
```

使用时可以通过 `squeeze` 或切片去掉：
```python
image = obs['wrist_1'][0]  # (128, 128, 3)
```

### Q2: side_policy 和 side_classifier 有什么区别？

**A**: 它们可能指向**同一个物理相机**，但：
- **side_policy**: 用于训练策略网络（actor-critic）
- **side_classifier**: 用于训练奖励分类器

两者可能使用不同的：
- 裁剪区域（不同的感兴趣区域）
- 数据增强策略
- 输入归一化方法

### Q3: 如何将图像用于神经网络输入？

**A**: 需要进行以下转换：
```python
# 1. 去掉 batch 维度
image = obs['wrist_1'][0]  # (128, 128, 3)

# 2. 转换数据类型并归一化
image = image.astype(np.float32) / 255.0  # [0, 1]

# 3. 转置为 CHW 格式（如果使用 PyTorch）
image = np.transpose(image, (2, 0, 1))  # (3, 128, 128)

# 4. 转换为 Tensor
import torch
image_tensor = torch.from_numpy(image)
```

### Q4: 为什么要分 buffer 和 demo_buffer？

**A**: 
- **buffer**: 包含所有数据（Policy 生成 + 人工演示）
- **demo_buffer**: 仅包含人工演示

这样设计的原因：
1. 训练时可以对人工演示施加更高的权重
2. 可以单独分析 Policy 和人工的性能
3. 避免 Policy 早期探索的低质量数据污染演示

### Q5: grasp_penalty 是什么？

**A**: 这是任务相关的惩罚项，通常用于：
- 防止夹爪过早关闭
- 惩罚不必要的夹爪动作
- 塑造更平滑的控制策略

具体含义取决于任务的奖励函数设计。

---

## 相关工具

### 分析工具

1. **analyze_policy_jitter.py**: 分析动作抖动和 episode 统计
   ```bash
   python utils/analyze_policy_jitter.py
   python utils/analyze_policy_jitter.py --last-n-files 5
   ```

2. **可视化工具**: 查看 buffer 中的图像和轨迹
   ```python
   # 保存图像示例
   import cv2
   for i, t in enumerate(transitions[:10]):
       img = t['observations']['wrist_1'][0]
       cv2.imwrite(f'frame_{i}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
   ```

### 数据统计

```bash
# 统计文件数量和大小
ls -lh checkpoints/buffer/ | wc -l
du -sh checkpoints/buffer/
du -sh checkpoints/demo_buffer/

# 快速查看最新文件
ls -lt checkpoints/buffer/*.pkl | head -5
```

---

## 总结

HIL-SERL 的 replay buffer 采用了标准的 **off-policy RL** 数据格式：

- **存储格式**: Pickle 序列化的 Python list
- **数据单元**: Transition (s, a, s', r, done)
- **观测空间**: 多模态（4 个相机图像 + 19 维状态）
- **动作空间**: 7 维连续动作 (6D 末端增量 + 1D 夹爪)
- **奖励信号**: 二值奖励（learned classifier）

这种格式：
- ✅ 简单直观，易于理解和调试
- ✅ 包含完整的多模态信息
- ✅ 支持离线训练和在线更新
- ⚠️ 文件较大（主要是图像数据）
- ⚠️ 需要手动管理 buffer 大小和旧文件清理
