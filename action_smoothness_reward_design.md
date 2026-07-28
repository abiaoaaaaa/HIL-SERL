# 动作平滑惩罚设计方案

## 1. 问题分析

**现象**：关节跳变抖动比较大  
**原因**：策略网络输出的连续两帧动作可能有较大差异，导致关节加速度过大  
**目标**：通过奖励函数中的惩罚项，鼓励策略输出平滑的动作序列

---

## 2. 设计方案

### 2.1 动作平滑惩罚项

在 reward 函数中添加一个惩罚项，惩罚连续两帧动作的 L2 距离：

```python
action_penalty = λ * ||action_t - action_{t-1}||₂²
```

其中：
- `action_t`: 当前时刻的动作
- `action_{t-1}`: 上一时刻的动作
- `λ`: 惩罚权重系数（建议范围: 0.01 ~ 0.1）

### 2.2 完整 Reward 函数设计

采用连续 reward（而非二值 reward），包含三个部分：

```python
reward = r_task + r_safety - r_action_penalty

其中:
r_task      = 任务进度奖励（接近目标）
r_safety    = 安全惩罚（越界、碰撞等）
r_action_penalty = 动作平滑惩罚
```

---

## 3. 具体实现

### 3.1 修改 MarvinEnv.__init__

在 `__init__` 中初始化 `last_action`：

```python
# 在 line 218 附近
self.last_action = np.zeros(5, dtype=np.float32)  # [dx,dy,dz,dry,gripper]
```

✅ **已存在，无需修改**

### 3.2 修改 step() 方法

在 step 中保存当前动作（供下一帧使用）：

```python
# 在 line 937 附近，action clip 之后
self.last_action = action.copy()
```

✅ **已存在，无需修改**

### 3.3 修改 compute_reward() 方法

**方案 A：保持二值 reward，添加 info 返回惩罚值**

```python
def compute_reward(self, obs, action) -> tuple:
    """
    计算 reward 和 info

    Returns:
        (reward, info_dict)
    """
    current_pose = obs["state"]["tcp_pose"]  # [x,y,z,qx,qy,qz,qw] (m, quat)

    # 1. 位置误差 (mm)
    delta_xyz = np.abs(current_pose[:3] * 1000.0 - self._TARGET_POSE[:3])

    # 2. 旋转误差
    current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
    target_rot = Rotation.from_euler("xyz", np.deg2rad(self._TARGET_POSE[3:])).as_matrix()
    diff_rot = current_rot.T @ target_rot
    diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
    delta_rot = np.abs(np.rad2deg(diff_euler))

    # 3. 任务完成判断（二值）
    task_success = (
        np.all(delta_xyz < self._REWARD_THRESHOLD[:3]) and
        np.all(delta_rot < self._REWARD_THRESHOLD[3:])
    )

    # 4. 动作平滑惩罚（仅位置+旋转，不包括夹爪）
    action_diff = action[:4] - self.last_action[:4]  # [dx,dy,dz,dry]
    action_penalty = np.sum(action_diff ** 2)  # L2 平方

    # 5. 返回 info（包含所有调试信息）
    info = {
        'delta_xyz_mm': delta_xyz,
        'delta_rot_deg': delta_rot,
        'action_penalty': action_penalty,
        'task_success': task_success,
    }

    return float(task_success), info
```

**方案 B：连续 reward（推荐）**

```python
def compute_reward(self, obs, action) -> tuple:
    """
    计算连续 reward 和 info

    Reward 组成:
        r_total = r_task - λ_action * r_action - λ_safety * r_safety

    其中:
        r_task: 距离目标的负距离（越近越大）
        r_action: 动作跳变惩罚（L2 距离）
        r_safety: 安全惩罚（越界、接近边界）

    Returns:
        (reward, info_dict)
    """
    current_pose = obs["state"]["tcp_pose"]  # [x,y,z,qx,qy,qz,qw] (m, quat)

    # ==================== 1. 任务进度奖励 ====================
    # 位置误差 (m)
    delta_xyz_m = np.abs(current_pose[:3] - self._TARGET_POSE[:3] / 1000.0)
    pos_error = np.linalg.norm(delta_xyz_m)

    # 旋转误差 (rad)
    current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
    target_rot = Rotation.from_euler("xyz", np.deg2rad(self._TARGET_POSE[3:])).as_matrix()
    diff_rot = current_rot.T @ target_rot
    diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
    rot_error = np.linalg.norm(diff_euler)

    # 任务奖励 = 负加权距离（越近越大）
    r_task = -(10.0 * pos_error + 1.0 * rot_error)

    # ==================== 2. 动作平滑惩罚 ====================
    # 只对位置+旋转（不包括夹爪）计算跳变
    action_diff = action[:4] - self.last_action[:4]  # [dx,dy,dz,dry]
    r_action = np.sum(action_diff ** 2)  # L2 平方

    # ==================== 3. 安全惩罚 ====================
    # 检查是否接近边界（使用配置的 safety box）
    xyz_m = current_pose[:3]
    xyz_low = self.xyz_bounding_box.low
    xyz_high = self.xyz_bounding_box.high

    # 距离边界的最小距离
    dist_to_boundary = np.min([
        xyz_m - xyz_low,
        xyz_high - xyz_m
    ])

    # 如果距离边界 < 5mm，开始惩罚
    BOUNDARY_MARGIN = 0.005  # 5mm
    if dist_to_boundary < BOUNDARY_MARGIN:
        r_safety = 10.0 * (BOUNDARY_MARGIN - dist_to_boundary) ** 2
    else:
        r_safety = 0.0

    # ==================== 4. 组合 Reward ====================
    LAMBDA_ACTION = 0.05   # 动作惩罚权重（可调）
    LAMBDA_SAFETY = 1.0    # 安全惩罚权重

    reward = r_task - LAMBDA_ACTION * r_action - LAMBDA_SAFETY * r_safety

    # ==================== 5. 任务成功判断 ====================
    delta_xyz_mm = delta_xyz_m * 1000.0
    delta_rot_deg = np.abs(np.rad2deg(diff_euler))
    task_success = (
        np.all(delta_xyz_mm < self._REWARD_THRESHOLD[:3]) and
        np.all(delta_rot_deg < self._REWARD_THRESHOLD[3:])
    )

    # ==================== 6. Info ====================
    info = {
        'delta_xyz_mm': delta_xyz_mm,
        'delta_rot_deg': delta_rot_deg,
        'r_task': r_task,
        'r_action': r_action,
        'r_safety': r_safety,
        'reward_total': reward,
        'action_penalty': r_action,
        'task_success': task_success,
    }

    return reward, info
```

### 3.4 修改 step() 方法签名

由于 `compute_reward` 现在需要 `action` 参数，需要修改 step 中的调用：

```python
# 在 line 973 附近
# 修改前:
reward = self.compute_reward(obs)

# 修改后:
reward, reward_info = self.compute_reward(obs, action)

# 合并到 info
info.update(reward_info)
```

---

## 4. 超参数调优建议

### 4.1 动作惩罚权重 `LAMBDA_ACTION`

| 权重值 | 效果 |
|--------|------|
| 0.01   | 轻微惩罚，允许较大动作跳变 |
| 0.05   | **推荐起点**，平衡性能与平滑 |
| 0.1    | 较强惩罚，动作非常平滑但可能过于保守 |
| 0.5    | 极强惩罚，可能导致策略不敢动作 |

**调优方法**：
1. 从 `0.05` 开始训练 50k 步
2. 观察 `info['action_penalty']` 的平均值
3. 如果抖动仍明显，逐步增加到 `0.1`
4. 如果策略过于保守（不敢靠近目标），减小到 `0.02`

### 4.2 任务奖励权重

```python
r_task = -(w_pos * pos_error + w_rot * rot_error)
```

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `w_pos` | 10.0 | 位置权重（1mm = -0.01 reward） |
| `w_rot` | 1.0  | 旋转权重（1° ≈ 0.017 rad = -0.017 reward） |

**平衡原则**：位置误差 1mm 的惩罚 ≈ 旋转误差 0.57° 的惩罚

### 4.3 动态权重调整（进阶）

在训练初期使用较小的动作惩罚，后期逐渐增加：

```python
# 在 compute_reward 中
training_progress = self.curr_path_length / self.max_episode_length
LAMBDA_ACTION = 0.02 + 0.08 * training_progress  # 从 0.02 线性增加到 0.1
```

---

## 5. 实施步骤

### 步骤 1：添加配置参数

在 `marvin_env/config/xxx_config.py` 中添加：

```python
# Reward 配置
LAMBDA_ACTION = 0.05    # 动作平滑惩罚权重
LAMBDA_SAFETY = 1.0     # 安全惩罚权重
TASK_POS_WEIGHT = 10.0  # 位置误差权重
TASK_ROT_WEIGHT = 1.0   # 旋转误差权重
BOUNDARY_MARGIN = 0.005 # 边界安全裕度 (m)
```

### 步骤 2：修改 compute_reward

选择方案 B（连续 reward），实现上述代码。

### 步骤 3：修改 step 调用

```python
reward, reward_info = self.compute_reward(obs, action)
info.update(reward_info)
```

### 步骤 4：验证

在 actor loop 中打印 reward 分解：

```python
if done or self.curr_path_length % 10 == 0:
    print(f"[REWARD] r_total={reward:.3f} | "
          f"r_task={info['r_task']:.3f} | "
          f"r_action={info['r_action']:.4f} | "
          f"r_safety={info['r_safety']:.3f}")
```

### 步骤 5：训练监控

在 TensorBoard 中记录：
- `reward/total`
- `reward/task`
- `reward/action_penalty`
- `reward/safety`
- `metrics/action_diff_mean`

---

## 6. 预期效果

### 训练前（无动作惩罚）
```
action_diff_norm: 0.15 ~ 0.30  (较大跳变)
关节加速度峰值: 50 ~ 100 rad/s²
轨迹平滑度: 低
```

### 训练后（有动作惩罚）
```
action_diff_norm: 0.03 ~ 0.08  (平滑)
关节加速度峰值: 10 ~ 30 rad/s²
轨迹平滑度: 高
```

---

## 7. 潜在问题与解决

### 问题 1：策略过于保守

**症状**：机器人几乎不动，或移动非常缓慢  
**原因**：动作惩罚权重过大  
**解决**：减小 `LAMBDA_ACTION` 到 0.01 或更小

### 问题 2：仍有抖动

**症状**：平均动作跳变减小，但仍有偶发的大跳变  
**原因**：策略网络输出分布的标准差过大  
**解决**：
1. 增大 `LAMBDA_ACTION` 到 0.1
2. 在 SAC 中减小 action std 的初始化值
3. 使用 action filter（例如指数移动平均）

### 问题 3：训练不稳定

**症状**：reward 震荡，不收敛  
**原因**：多个 reward 项的量纲不匹配  
**解决**：标准化各项 reward 使其在 [-1, 1] 范围内

---

## 8. 代码修改清单

| 文件 | 行号 | 修改内容 |
|------|------|---------|
| `marvin_env.py` | 218 | ✅ 已有 `self.last_action` 初始化 |
| `marvin_env.py` | 937 | ✅ 已有 `self.last_action = action.copy()` |
| `marvin_env.py` | 1235-1258 | 🔧 修改 `compute_reward` 签名和实现 |
| `marvin_env.py` | 973 | 🔧 修改 step 中的 reward 调用 |
| `config/xxx_config.py` | 新增 | 添加 reward 权重配置 |

---

## 9. 参考文献

1. **Soft Actor-Critic**: 使用连续 reward 而非稀疏 reward
2. **DMControl**: 动作平滑惩罚的标准做法
3. **TD3**: action smoothing via target policy smoothing

---

**文档版本**: 1.0  
**创建日期**: 2026-07-27
