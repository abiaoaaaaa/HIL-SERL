# HIL-SERL 动作空间输入与转换分析文档

## 概述

本文档详细分析 HIL-SERL 项目中从动作输入（模型推理和空间鼠标）到控制命令发出的完整数据流。

---

## 1. 动作空间定义

### 1.1 基础动作空间
在 `franka_env.py` 中定义：

```python
self.action_space = gym.spaces.Box(
    np.ones((7,), dtype=np.float32) * -1,
    np.ones((7,), dtype=np.float32),
)
```

**维度说明**：
- **维度 0-2**: xyz 位置增量（归一化到 [-1, 1]）
- **维度 3-5**: 旋转向量（轴角表示，归一化到 [-1, 1]）
- **维度 6**: 夹爪动作（-1: 关闭, +1: 打开）

### 1.2 固定夹爪模式
使用 `GripperCloseEnv` wrapper 时，动作空间降维为 6D：

```python
self.action_space = Box(ub.low[:6], ub.high[:6])  # 只有位置和姿态
```

---

## 2. 动作输入来源

### 2.1 模型推理输入

#### 训练阶段（Actor Loop）
位置：`examples/train_rlpd.py` - `actor()` 函数

**推理流程**：
```python
# 1. 采样动作
sampling_rng, key = jax.random.split(sampling_rng)
actions = agent.sample_actions(
    observations=jax.device_put(obs),
    seed=key,
    argmax=False,  # 训练时使用随机采样
)
actions = np.asarray(jax.device_get(actions))

# 2. 执行环境步进
next_obs, reward, done, truncated, info = env.step(actions)
```

**Agent 采样方法**（`sac.py`）：
```python
@partial(jax.jit, static_argnames=("argmax",))
def sample_actions(
    self,
    observations: Data,
    *,
    seed: Optional[PRNGKey] = None,
    argmax: bool = False,
    **kwargs,
) -> jnp.ndarray:
    dist = self.forward_policy(observations, rng=seed, train=False)
    if argmax:
        return dist.mode()  # 评估时使用模式值
    else:
        return dist.sample(seed=seed)  # 训练时随机采样
```

#### 评估阶段
```python
actions = agent.sample_actions(
    observations=jax.device_put(obs),
    argmax=False,  # 或 True，取决于评估需求
    seed=key
)
```

**模型输出**：
- SAC Policy 网络输出高斯分布的均值和标准差
- 通过 `tanh_squash_distribution` 将输出映射到 [-1, 1]

### 2.2 空间鼠标输入

#### SpaceMouse 硬件接口
位置：`spacemouse_expert.py`

**持续读取线程**：
```python
def _read_spacemouse(self):
    while True:
        state = pyspacemouse.read_all()
        action = [0.0] * 6
        buttons = [0, 0, 0, 0]
        
        if len(state) == 1:  # 单臂
            action = [
                -state[0].y,      # x 方向
                state[0].x,       # y 方向
                state[0].z,       # z 方向
                -state[0].roll,   # roll
                -state[0].pitch,  # pitch
                -state[0].yaw     # yaw
            ]
            buttons = state[0].buttons
        
        # 更新共享状态
        self.latest_data["action"] = action
        self.latest_data["buttons"] = buttons
```

**获取动作**：
```python
def get_action(self) -> Tuple[np.ndarray, list]:
    action = self.latest_data["action"]
    buttons = self.latest_data["buttons"]
    return np.array(action), buttons
```

#### 人机干预 Wrapper
位置：`wrappers.py` - `SpacemouseIntervention`

**干预逻辑**：
```python
def action(self, action: np.ndarray) -> np.ndarray:
    expert_a, buttons = self.expert.get_action()
    self.left, self.right = tuple(buttons)
    intervened = False
    
    # 检查是否有空间鼠标输入
    if np.linalg.norm(expert_a) > 0.001:
        intervened = True
    
    # 处理夹爪按钮
    if self.gripper_enabled:
        if self.left:  # 关闭夹爪
            gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            intervened = True
        elif self.right:  # 打开夹爪
            gripper_action = np.random.uniform(0.9, 1, size=(1,))
            intervened = True
        else:
            gripper_action = np.zeros((1,))
        expert_a = np.concatenate((expert_a, gripper_action), axis=0)
    
    # 如果有干预，返回空间鼠标动作
    if intervened:
        return expert_a, True
    
    # 否则返回策略动作
    return action, False
```

**在 step 中应用**：
```python
def step(self, action):
    new_action, replaced = self.action(action)
    obs, rew, done, truncated, info = self.env.step(new_action)
    
    if replaced:
        info["intervene_action"] = new_action  # 标记为干预动作
    
    return obs, rew, done, truncated, info
```

---

## 3. 动作转换流程

### 3.1 动作缩放（Action Scaling）
位置：`franka_env.py` - `step()` 方法

**配置参数**（以 RAM Insertion 为例）：
```python
ACTION_SCALE = (0.01, 0.06, 1)
# [0]: xyz 位置缩放因子 (m)
# [1]: 旋转缩放因子 (rad)
# [2]: 夹爪缩放因子
```

**缩放过程**：
```python
def step(self, action: np.ndarray) -> tuple:
    # 1. 裁剪到动作空间范围
    action = np.clip(action, self.action_space.low, self.action_space.high)
    
    # 2. 提取 xyz 增量
    xyz_delta = action[:3]
    
    # 3. 计算新位置（位置增量）
    self.nextpos = self.currpos.copy()
    self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
    
    # 4. 计算新姿态（旋转增量）
    self.nextpos[3:] = (
        Rotation.from_rotvec(action[3:6] * self.action_scale[1])
        * Rotation.from_quat(self.currpos[3:])
    ).as_quat()
    
    # 5. 提取夹爪动作
    gripper_action = action[6] * self.action_scale[2]
```

### 3.2 安全边界裁剪
```python
def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
    # 裁剪 xyz 位置
    pose[:3] = np.clip(
        pose[:3], 
        self.xyz_bounding_box.low, 
        self.xyz_bounding_box.high
    )
    
    # 转换为欧拉角并裁剪
    euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
    
    # 处理 roll 角的不连续性（pi 到 -pi）
    sign = np.sign(euler[0])
    euler[0] = sign * np.clip(
        np.abs(euler[0]),
        self.rpy_bounding_box.low[0],
        self.rpy_bounding_box.high[0],
    )
    
    # 裁剪 pitch 和 yaw
    euler[1:] = np.clip(
        euler[1:], 
        self.rpy_bounding_box.low[1:], 
        self.rpy_bounding_box.high[1:]
    )
    
    # 转换回四元数
    pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
    
    return pose
```

**边界定义**（RAM Insertion 示例）：
```python
TARGET_POSE = np.array([0.588, -0.036, 0.278, π, 0, 0])
ABS_POSE_LIMIT_LOW = TARGET_POSE - [0.03, 0.02, 0.01, 0.01, 0.1, 0.4]
ABS_POSE_LIMIT_HIGH = TARGET_POSE + [0.03, 0.02, 0.05, 0.01, 0.1, 0.4]
```

### 3.3 夹爪命令处理
```python
def _send_gripper_command(self, pos: float, mode="binary"):
    if mode == "binary":
        # 关闭夹爪条件
        if (pos <= -0.5) and \
           (self.curr_gripper_pos > 0.85) and \
           (time.time() - self.last_gripper_act > self.gripper_sleep):
            requests.post(self.url + "close_gripper")
            self.last_gripper_act = time.time()
            time.sleep(self.gripper_sleep)  # 默认 0.6s
        
        # 打开夹爪条件
        elif (pos >= 0.5) and \
             (self.curr_gripper_pos < 0.85) and \
             (time.time() - self.last_gripper_act > self.gripper_sleep):
            requests.post(self.url + "open_gripper")
            self.last_gripper_act = time.time()
            time.sleep(self.gripper_sleep)
```

---

## 4. 控制命令发送

### 4.1 位置命令发送
位置：`franka_env.py`

```python
def _send_pos_command(self, pos: np.ndarray):
    """发送位置命令到机器人服务器"""
    self._recover()  # 清除错误
    arr = np.array(pos).astype(np.float32)
    data = {"arr": arr.tolist()}
    requests.post(self.url + "pose", json=data)
```

**命令格式**：
- 7维数组：[x, y, z, qx, qy, qz, qw]
- 单位：米（位置），四元数（姿态）

### 4.2 机器人服务器处理
位置：`robot_servers/franka_server.py`

**Flask 路由**：
```python
@webapp.route("/pose", methods=["POST"])
def pose():
    pos = np.array(request.json["arr"])
    robot_server.move(pos)
    return "Moved"
```

**ROS 发布**：
```python
def move(self, pose: list):
    """移动到目标位姿: [x, y, z, qx, qy, qz, qw]"""
    assert len(pose) == 7
    msg = geom_msg.PoseStamped()
    msg.header.frame_id = "0"
    msg.header.stamp = rospy.Time.now()
    msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
    msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
    
    # 发布到阻抗控制器
    self.eepub.publish(msg)
```

**ROS Topic**：
```
/cartesian_impedance_controller/equilibrium_pose
```

### 4.3 控制频率管理
```python
def step(self, action: np.ndarray) -> tuple:
    start_time = time.time()
    
    # ... 动作处理和命令发送 ...
    
    # 控制循环频率
    dt = time.time() - start_time
    time.sleep(max(0, (1.0 / self.hz) - dt))
```

**默认频率**：10 Hz（在 `__init__` 中设置 `hz=10`）

---

## 5. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        动作输入源                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼────────┐         ┌───────▼────────┐
        │  模型推理输入   │         │ 空间鼠标输入    │
        │                │         │                │
        │ SAC Policy     │         │ SpaceMouse     │
        │ ↓              │         │ Expert         │
        │ Tanh Squash    │         │ ↓              │
        │ ↓              │         │ 6D动作+按钮    │
        │ [-1, 1]^7      │         │ [-1, 1]^6 + 按钮│
        └───────┬────────┘         └───────┬────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Spacemouse        │
                    │ Intervention      │
                    │ Wrapper           │
                    │                   │
                    │ 干预判断：         │
                    │ ‖action‖>0.001?  │
                    │ 按钮按下?         │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ 选择最终动作       │
                    │ 干预 → 空间鼠标    │
                    │ 否则 → 策略动作    │
                    └─────────┬─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                     FrankaEnv.step()                      │
│                                                           │
│  1. 动作裁剪: clip(action, -1, 1)                         │
│                                                           │
│  2. 位置增量计算:                                          │
│     xyz_new = xyz_curr + action[:3] * scale_pos         │
│                                                           │
│  3. 姿态增量计算:                                          │
│     R_delta = Rotation.from_rotvec(action[3:6] * scale_rot) │
│     quat_new = R_delta * quat_curr                       │
│                                                           │
│  4. 安全边界裁剪:                                          │
│     pose = clip_safety_box([xyz_new, quat_new])         │
│                                                           │
│  5. 夹爪命令处理:                                          │
│     if action[6] < -0.5: close_gripper()                │
│     elif action[6] > 0.5: open_gripper()                │
└─────────────────────────────┬─────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
      ┌─────────▼─────────┐       ┌────────▼────────┐
      │ _send_pos_command │       │ _send_gripper   │
      │                   │       │ _command        │
      │ HTTP POST         │       │                 │
      │ /pose             │       │ HTTP POST       │
      │                   │       │ /close_gripper  │
      │ [x,y,z,qx,qy,qz,qw]│      │ /open_gripper   │
      └─────────┬─────────┘       └────────┬────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                   Franka Server (Flask)                   │
│                                                           │
│  HTTP → ROS 转换                                          │
│                                                           │
│  @webapp.route("/pose")                                   │
│  def pose():                                              │
│      robot_server.move(pose)                             │
│                                                           │
│  def move(pose):                                          │
│      msg = PoseStamped()                                 │
│      msg.pose.position = Point(x, y, z)                  │
│      msg.pose.orientation = Quaternion(qx,qy,qz,qw)      │
│      self.eepub.publish(msg)                             │
└─────────────────────────────┬─────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   ROS Topic       │
                    │   /cartesian_     │
                    │   impedance_      │
                    │   controller/     │
                    │   equilibrium_pose│
                    └─────────┬─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│          Franka Impedance Controller (ROS)                │
│                                                           │
│  serl_franka_controllers                                  │
│  阻抗控制器                                                │
│  ↓                                                        │
│  关节扭矩计算                                              │
│  ↓                                                        │
│  发送到 Franka 控制器                                      │
└─────────────────────────────┬─────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Franka Robot     │
                    │  执行运动          │
                    └───────────────────┘
```

---

## 6. 关键时序参数

### 6.1 控制循环
- **环境步频**: 10 Hz（100ms/步）
- **夹爪延迟**: 0.6s（GRIPPER_SLEEP）
- **动作重试间隔**: 基于频率自动计算

### 6.2 空间鼠标
- **读取频率**: 持续循环（独立进程）
- **干预阈值**: ‖action‖ > 0.001
- **按钮响应**: 即时

### 6.3 网络同步
- **参数更新频率**: 每 50 步（steps_per_update）
- **日志周期**: 每 10 步
- **Checkpoint周期**: 每 5000 步（任务相关）

---

## 7. 坐标系与参考框架

### 7.1 相对框架转换
使用 `RelativeFrame` wrapper（`relative_env.py`）：

```python
# 动作空间：相对于当前末端执行器的增量
# 观测空间：相对于重置位置的位姿
```

### 7.2 姿态表示转换

**内部表示**：四元数 [qx, qy, qz, qw]

**对外接口**：
- 策略输入/输出：欧拉角 [roll, pitch, yaw]（通过 `Quat2EulerWrapper`）
- 控制命令：四元数

**转换流程**：
```
策略输出(欧拉角) → 转四元数 → 增量旋转 → 裁剪 → 四元数命令 → 机器人
```

---

## 8. 示例：RAM Insertion 任务

### 8.1 配置参数
```python
ACTION_SCALE = (0.01, 0.06, 1)  # 1cm位置, 3.4°旋转
TARGET_POSE = [0.588, -0.036, 0.278, π, 0, 0]
RESET_POSE = TARGET_POSE + [0, 0, 0.05, 0, 0.05, 0]
MAX_EPISODE_LENGTH = 100
```

### 8.2 动作转换示例

**输入动作**: `[0.5, 0.0, -0.3, 0.1, 0.0, 0.0]` (6D, 固定夹爪)

**转换步骤**：
1. **缩放**:
   - xyz: [0.5, 0.0, -0.3] * 0.01 = [0.005, 0.0, -0.003] m
   - rot: [0.1, 0.0, 0.0] * 0.06 = [0.006, 0.0, 0.0] rad (≈0.34°)

2. **当前位姿**: [0.590, -0.036, 0.280, quat_curr]

3. **计算新位姿**:
   - xyz_new = [0.595, -0.036, 0.277]
   - quat_new = Rotation.from_rotvec([0.006, 0, 0]) * quat_curr

4. **边界检查**: 确保在 [LOW, HIGH] 范围内

5. **发送命令**: HTTP POST → ROS → 机器人

---

## 9. 训练中的动作流

### 9.1 数据收集
```python
# Actor 循环
action_policy = agent.sample_actions(obs)  # 策略采样
action_space_mouse, intervened = spacemouse.action(action_policy)
action_final = action_space_mouse if intervened else action_policy

# 执行并记录
next_obs, reward, done, info = env.step(action_final)
transition = {
    'actions': action_final,  # 记录实际执行的动作
    'observations': obs,
    'next_observations': next_obs,
    ...
}

# 干预动作单独存储
if intervened:
    intvn_data_store.insert(transition)  # 干预数据（演示质量）
data_store.insert(transition)  # 所有数据
```

### 9.2 学习器采样
```python
# 50/50 采样（RLPD）
replay_batch = replay_buffer.sample(batch_size=128)  # 在线经验
demo_batch = demo_buffer.sample(batch_size=128)      # 演示+干预
batch = concat_batches(replay_batch, demo_batch)

# 训练
agent, info = agent.update(batch)
```

---

## 10. 总结

### 动作流关键特性：
1. **双输入融合**: 模型推理与人类干预无缝切换
2. **多层转换**: 归一化 → 缩放 → 增量 → 边界裁剪
3. **安全机制**: 速度限制、边界框、夹爪防抖
4. **分层架构**: Python(Gym) → HTTP → ROS → 机器人控制器
5. **异步设计**: 空间鼠标独立进程，Actor-Learner 分离

### 性能考虑：
- HTTP 通信延迟: ~5-10ms
- 控制频率: 10Hz 足够精确操作
- JAX JIT 编译: 模型推理 <5ms
- 空间鼠标: 零延迟人类干预

---

## 11. 动作增量到末端新位姿的详细过程

### 11.1 动作输入与归一化

在 `franka_env.py` 的 `step()` 方法中：

```python
action = np.clip(action, self.action_space.low, self.action_space.high)
# action_space: [-1, 1]^7
```

- 策略网络输出的动作被限制在 `[-1, 1]` 范围
- 此时的动作是**归一化的增量**，还未转换到实际的物理单位

### 11.2 坐标系转换链（以USB插入任务为例）

动作经过以下转换链：

#### 步骤A：末端执行器坐标系 → 基座坐标系

**RelativeFrame wrapper** (`relative_env.py: 91-98`):

```python
def transform_action(self, action: np.ndarray):
    action = np.array(action)
    action[:6] = self.transform_matrix @ action[:6]
    return action
```

**变换矩阵的构造** (`transformations.py: 26-36`):

```python
def construct_transform_matrix(tcp_pose):
    """
    构造从末端坐标系到基座坐标系的变换矩阵
    用于delta pose控制的机器人（如Franka）
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()  # 提取3x3旋转矩阵
    transform_matrix = np.zeros((6, 6))
    transform_matrix[:3, :3] = rotation    # 左上角：位置增量的旋转
    transform_matrix[3:, 3:] = rotation    # 右下角：旋转增量的旋转
    return transform_matrix
```

**矩阵形式**：
```
T = [R  0]
    [0  R]  (6×6)
```

其中 `R` 是从当前末端位姿的四元数提取的 3×3 旋转矩阵。

**关键点**：
- 使用**分块对角旋转矩阵**，而非伴随矩阵（Adjoint）
- 因为 Franka 使用的是**delta pose 控制**，而非 twist 控制
- 左下角没有耦合项（`skew(p)@R`）

#### 步骤B：基座坐标系下的位置更新

```python
xyz_delta = action[:3]
self.nextpos = self.currpos.copy()
self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]
```

对于 USB 任务：
- `ACTION_SCALE[0] = 0.015` (米)
- 归一化动作 `[-1, 1]` 被缩放到最大 `±15mm` 的位置增量
- 位置更新：`new_xyz = current_xyz + action[:3] * 0.015`

#### 步骤C：基座坐标系下的旋转更新

```python
self.nextpos[3:] = (
    Rotation.from_rotvec(action[3:6] * self.action_scale[1])
    * Rotation.from_quat(self.currpos[3:])
).as_quat()
```

对于 USB 任务：
- `ACTION_SCALE[1] = 0.1` (弧度)
- 归一化动作 `[-1, 1]` 被缩放到最大 `±0.1 rad ≈ ±5.7°` 的旋转增量
- 旋转更新：采用**旋转向量**（axis-angle）形式，然后转换为四元数进行左乘

**旋转组合公式**：
```
q_new = q_delta * q_current
```
其中 `q_delta = exp(action[3:6] * 0.1)`

#### 步骤D：安全边界裁剪

```python
self.nextpos = self.clip_safety_box(self.nextpos)
```

对于 USB 任务的安全边界：
```python
TARGET_POSE = [0.553, 0.177, 0.251, π, 0, -π/2]
ABS_POSE_LIMIT_HIGH = TARGET_POSE + [0.03, 0.06, 0.05, 0.1, 0.1, 0.3]
ABS_POSE_LIMIT_LOW = TARGET_POSE - [0.03, 0.01, 0.03, 0.1, 0.1, 0.3]
```

工作空间：
- x 方向：`[0.523, 0.583]` (60mm 范围)
- y 方向：`[0.167, 0.237]` (70mm 范围)
- z 方向：`[0.221, 0.301]` (80mm 范围)
- 旋转范围：约 `±6°` (roll/pitch), `±17°` (yaw)

#### 步骤E：发送到 Franka 控制器

```python
self._send_pos_command(self.clip_safety_box(self.nextpos))
```

发送的位姿格式：`[x, y, z, qx, qy, qz, qw]` (7 维)，单位为米和四元数。

### 11.3 完整数据流总结

```
1. 策略网络输出归一化动作 a ∈ [-1,1]^7 (末端坐标系)
   ↓
2. RelativeFrame.transform_action: 
   a[:6] = R_current @ a[:6] (转到基座坐标系)
   ↓
3. FrankaEnv.step:
   - 位置增量: Δxyz = a[:3] * 0.015
   - 旋转增量: Δq = exp(a[3:6] * 0.1)
   - 新位置: xyz_new = xyz_current + Δxyz
   - 新旋转: q_new = Δq * q_current
   ↓
4. 安全裁剪: clip_safety_box(nextpos)
   ↓
5. 发送到机器人: POST /pose {[x,y,z,qx,qy,qz,qw]}
   ↓
6. Franka 控制器: 阻抗控制实现目标位姿
   ↓
7. 读取新状态: getstate() → currpos (基座坐标系)
   ↓
8. RelativeFrame.transform_observation:
   - tcp_pose = T_r_o_inv @ currpos (转到重置相对系)
   - tcp_vel = R_current^T @ vel (转到末端坐标系)
   ↓
9. 返回给策略: obs["state"]["tcp_pose"] (重置相对系)
```

---

## 12. 坐标系与参考框架详解

### 12.1 基座坐标系（Base Frame / World Frame）

- **原点**：机器人底座
- **用途**：所有物理位置的绝对参考
- **在代码中**：
  - `currpos`：当前末端位姿（基座坐标系）
  - `nextpos`：下一个目标位姿（基座坐标系）
  - `TARGET_POSE`：目标插入位置（基座坐标系）

### 12.2 末端执行器坐标系（End-Effector Frame）

- **原点**：工具中心点（TCP）
- **方向**：随末端执行器旋转而变化
- **用途**：策略网络输出的动作在此坐标系下表达

**为什么使用末端坐标系？**
- 机器人学习时，"向前移动"的含义应该相对于夹爪当前朝向，而非固定的世界方向
- 这使得策略具有**旋转等变性**

### 12.3 重置位姿相对坐标系（Reset Pose Relative Frame）

对于 USB 任务：
```python
RESET_POSE = TARGET_POSE + [0, 0.03, 0.05, 0, 0, 0]
```

在 `RelativeFrame` wrapper 中：
```python
self.T_r_o_inv = np.linalg.inv(
    construct_homogeneous_matrix(obs["state"]["tcp_pose"])
)
```

#### 观测空间的相对位姿计算

```python
if self.include_relative_pose:
    T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])  # 当前位姿（基座系）
    T_b_r = self.T_r_o_inv @ T_b_o  # 相对于重置位姿的位姿
    
    p_b_r = T_b_r[:3, 3]  # 位置差
    theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()  # 旋转差
    obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))
```

**数学表达**：
```
T_reset_to_base^(-1) = T_r_o_inv (在 reset 时计算并固定)
T_current_to_base = T_b_o (每步更新)
T_current_to_reset = T_r_o_inv @ T_b_o
```

**观测到的 tcp_pose 含义**：
- **位置部分**：当前 TCP 相对于重置位置的位移向量（基座坐标系表达）
- **旋转部分**：当前 TCP 相对于重置姿态的旋转差（四元数）

对于 USB 任务：
- 重置位置在目标上方 30mm、前方 50mm
- 策略看到的位置观测 `[0, 0, 0]` 表示在重置位置
- 策略看到的位置观测 `[0, -0.03, -0.05]` 表示在目标位置

### 12.4 速度观测的坐标系

```python
transform_inv = np.linalg.inv(self.transform_matrix)
obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]
```

- 速度从**基座坐标系**转换到**当前末端执行器坐标系**
- 这样策略感知到的是"相对于自身的运动速度"

### 12.5 变换矩阵的构造与使用

#### 变换矩阵的来源

从机器人控制器返回的当前末端位姿的四元数部分提取：

```python
def _update_currpos(self):
    ps = requests.post(self.url + "getstate").json()
    self.currpos = np.array(ps["pose"])  # [x, y, z, qx, qy, qz, qw]

# 在 RelativeFrame 中
self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
```

#### 变换矩阵的更新时机

```python
def step(self, action: np.ndarray):
    # 1. 先用旧的变换矩阵转换动作
    transformed_action = self.transform_action(action)
    
    # 2. 执行动作
    obs, reward, done, truncated, info = self.env.step(transformed_action)
    
    # 3. 用新的末端位姿更新变换矩阵（为下一步准备）
    self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
    
    # 4. 转换观测
    transformed_obs = self.transform_observation(obs)
    return transformed_obs, reward, done, truncated, info
```

**关键点**：
- 在每个 step 开始时，`transform_matrix` 保存的是**上一步末端的姿态**
- 用这个矩阵转换当前动作
- step 结束后，更新矩阵为**当前末端的姿态**

#### 旋转矩阵的物理意义

假设当前末端执行器的姿态四元数为 `q = [qx, qy, qz, qw]`，对应的旋转矩阵为：

```
R_base_to_ee = [r11  r12  r13]
               [r21  r22  r23]
               [r31  r32  r33]
```

这个矩阵的**列向量**表示：
- 第 1 列：末端坐标系 x 轴在基座坐标系中的方向
- 第 2 列：末端坐标系 y 轴在基座坐标系中的方向  
- 第 3 列：末端坐标系 z 轴在基座坐标系中的方向

#### 位置增量的转换

如果策略输出的位置增量是 `Δp_ee = [Δx_ee, Δy_ee, Δz_ee]`（末端坐标系），要转换到基座坐标系：

```
Δp_base = R_base_to_ee @ Δp_ee
```

**示例**：USB 任务中末端姿态为 `[π, 0, -π/2]`（欧拉角）

转换为旋转矩阵（简化表示）：
```
R ≈ [0  -1   0]   # 末端 x 轴指向基座 -y 方向
    [0   0  -1]   # 末端 y 轴指向基座 -z 方向
    [1   0   0]   # 末端 z 轴指向基座 +x 方向
```

如果策略输出 `Δp_ee = [0.01, 0, 0]`（末端坐标系向前 10mm）：
```
Δp_base = R @ [0.01, 0, 0]ᵀ = [0, 0, 0.01]ᵀ
```
在基座坐标系中表现为沿 +x 方向移动 10mm。

#### 旋转增量的转换

旋转增量以**旋转向量**（axis-angle）形式表示：`ω_ee = [ωx, ωy, ωz]`

转换到基座坐标系：
```
ω_base = R_base_to_ee @ ω_ee
```

**为什么旋转也需要转换？**
- 旋转向量的方向定义了旋转轴
- 旋转轴在不同坐标系中的表示不同
- 例如：末端坐标系的"绕 z 轴旋转"对应基座坐标系的"绕某个倾斜轴旋转"

#### 为什么是分块对角矩阵？

变换矩阵的形式：
```
[R  0] [Δp]   [R@Δp]
[0  R] [Δω] = [R@Δω]
```

这是因为 Franka 使用的是**Delta Pose 控制**：
- 控制输入是位姿增量 `[Δx, Δy, Δz, Δrx, Δry, Δrz]`
- 位置和旋转是**解耦的**，分别进行坐标变换
- 不涉及平移引起的速度耦合项（这在 twist 控制中才需要）

#### 对比：伴随矩阵（Adjoint Matrix）

如果使用 twist 控制，需要用伴随矩阵：

```python
def construct_adjoint_matrix(tcp_pose):
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = [[0, -tz, ty],
                   [tz, 0, -tx],
                   [-ty, tx, 0]]
    
    adjoint_matrix = [[R,           0],
                      [skew@R,      R]]  # 注意左下角的耦合项
    return adjoint_matrix
```

**区别**：
- Adjoint 矩阵左下角有 `skew(p)@R` 项，表示平移和角速度的耦合
- 用于转换 twist（速度+角速度）
- Franka 的 delta pose 控制不需要这个耦合项

### 12.6 姿态的坐标系对齐分析

不同的初始姿态会导致末端坐标系与基座坐标系的不同对齐关系。

**示例：姿态 `[-90°, 0°, -90°]` 的对齐**

```python
euler = np.array([-90, 0, -90]) * np.pi / 180
R = R.from_euler('xyz', euler).as_matrix()

# 结果（近似）：
# 末端 x 轴 → 基座 -y 轴
# 末端 y 轴 → 基座 -z 轴
# 末端 z 轴 → 基座 +x 轴
```

**旋转映射关系**：

| 末端坐标系旋转 | 基坐标系效果 |
|--------------|------------|
| `rx_ee = 1` (绕末端 x 轴) | `ry_base = -1` (绕基座 y 轴) |
| `ry_ee = 1` (绕末端 y 轴) | `rz_base = -1` (绕基座 z 轴) |
| `rz_ee = 1` (绕末端 z 轴) | `rx_base = +1` (绕基座 x 轴) |

**关键发现**：在这个姿态下，**末端的 rz 旋转会影响基座的 x 轴旋转，而不是 z 轴**！

**z 轴对齐的姿态**：

如果希望末端 z 轴与基座 z 轴对齐（`rz_ee` 直接对应 `rz_base`），可以使用以下姿态：

| 姿态 (roll, pitch, yaw) | 末端 z 轴方向 | 系数 |
|----------------------|--------------|------|
| `[0°, 0°, 0°]` | 基座 +z | +1.000 |
| `[0°, 0°, 90°]` | 基座 +z | +1.000 |
| `[0°, 0°, -90°]` | 基座 +z | +1.000 |
| `[180°, 0°, 0°]` | 基座 -z | -1.000 |
| `[0°, 180°, 0°]` | 基座 -z | -1.000 |

### 12.7 关键理解总结

1. **动作空间**：末端执行器坐标系
2. **观测空间（位姿）**：相对于重置位置的增量（基座坐标系表达）
3. **观测空间（速度）**：末端执行器坐标系
4. **控制命令**：基座坐标系的绝对位姿
5. **变换矩阵来源**：从机器人控制器返回的当前末端位姿的四元数部分提取
6. **更新频率**：每个 step 后更新一次，用于下一步的动作转换
7. **矩阵形式**：分块对角 `[R, 0; 0, R]`，因为使用 delta pose 控制
8. **物理意义**：将末端坐标系中的增量向量旋转到基座坐标系
9. **坐标系对齐**：初始姿态决定了末端轴与基座轴的对齐关系

这种设计使得策略学习到的是**相对运动策略**，具有更好的泛化能力和物理直观性。

---

## 13. Marvin 5D 动作空间：6D → 5D 降维设计与 Bug 修复

### 13.1 动机

Marvin 机械臂的任务（平面操作）只需要控制基座坐标系的 xyz 位移 + z 轴旋转。
原有的 6D 动作空间 (dx, dy, dz, rx, ry, rz) 中，rx 和 ry 在 MarvinEnv 中被硬锁定到 RESET_POSE。
为了减少策略网络的输出维度，将动作空间从 6D (含 3 个旋转) 降维到 5D (仅 1 个 Z 旋转)：

```
6D (Franka): [dx, dy, dz, rx, ry, rz, gripper]  → 7 维
5D (Marvin): [dx, dy, dz, drz,     gripper]  → 5 维
```

### 13.2 Marvin RESET_POSE 与坐标系对齐分析

Marvin 的 RESET_POSE 为 `[-90.05°, 0.01°, -90.01°]` (roll, pitch, yaw)。
在此姿态（以及任意 yaw 但 roll=-90°, pitch=0° 姿态）下，存在关键的轴对齐关系：

```
EE X 轴 → 基座 (随 yaw 变化)
EE Y 轴 → 基座 [0, 0, -1]  ← 恒等于基座 -Z (与 yaw 无关!)
EE Z 轴 → 基座 (随 yaw 变化)
```

**核心几何事实**：在 `roll=-90°, pitch=0°` 的约束下，无论 yaw 如何变化，
EE 的 Y 轴始终指向基座 -Z 方向。

数学表达：

```
R[:, 1] = [sin(yaw) * sin(0°) - cos(yaw) * cos(0°) * sin(-90°), ... ]ᵀ
        = [0, 0, -1]ᵀ  (对任意 yaw 成立)
```

因此：
- **EE 的 ry 旋转 = 绕 EE Y 轴旋转 = 绕 基座 -Z 轴旋转**
- **EE 的 rx 旋转 = 绕 EE X 轴旋转 ≠ 绕 基座 Z 轴**
- **EE 的 rz 旋转 = 绕 EE Z 轴旋转 ≠ 绕 基座 Z 轴**

| yaw | EE Y → 基座 | `ry_ee=-0.1` → 基座 `rz` |
|-----|------------|-------------------------|
| -90° | `[0, 0, -1]` | `0.1` ✅ |
| 0° | `[0, 0, -1]` | `0.1` ✅ |
| 90° | `[0, 0, -1]` | `0.1` ✅ |

### 13.3 6D 参考实现 (6D + mask)

6D 时代的数据流（已验证正确）：

```
SpaceMouse 6D (EE系) [dx,dy,dz,rx,ry,rz]
  → SpacemouseIntervention: [dx,dy,dz,rx,ry,rz,gripper]  (7D, 直接拼接)
  → RelativeFrame.transform_action: action[:6] = T @ action[:6]  (全6维变换)
  → MarvinEnv._execute_sub_step:
      rot_base = action[3:6] * scale
      rot_base[0] = 0  # mask rx_base
      rot_base[1] = 0  # mask ry_base
      # rot_base[2] = rz_base, 保留
```

关键：6D 版本所有旋转分量都经过 `T @` 变换，然后在**基座系中 mask 掉 rx/ry**，只保留 rz。

### 13.4 SpaceMouse 维度语义

SpaceMouse 硬件原始输出（经过 `spacemouse_expert.py` 映射后）：

| 索引 | 表达式 | 语义 | 坐标系 |
|------|--------|------|--------|
| `expert_a[0]` | `+state.z` | dx (前后) | EE 系 |
| `expert_a[1]` | `+state.x` | dy (左右) | EE 系 |
| `expert_a[2]` | `+state.y` | dz (上下) | EE 系 |
| `expert_a[3]` | `-state.roll` | rx (绕 EE X) | EE 系 |
| `expert_a[4]` | `-state.pitch` | ry (绕 EE Y) | EE 系 |
| `expert_a[5]` | `-state.yaw` | rz (绕 EE Z) | EE 系 |

**人机交互直觉**：用户扭转 SpaceMouse 的 pitch 旋钮 → `expert_a[4]` 有值 → EE Y 轴旋转 → 基座 Z 轴旋转。

### 13.5 5D 降维 Bug 分析

#### Bug 所在：`relative_env.py` transform_action

降维到 5D 时，旋转分量从 `action[3]`（语义上对应 drz）需要填入 6D 向量的正确槽位：

```python
# 5D → pad 6D
action_6d = np.zeros(6)
action_6d[0] = action[0]  # dx
action_6d[1] = action[1]  # dy
action_6d[2] = action[2]  # dz
action_6d[5] = action[3]  # ← BUG: drz 填入了 EE rz 槽位!
# rx=0, ry=0 (保持不变)
```

**根因**：`action_6d[5]` 是 EE rz 槽位。EE Z 轴在 roll=-90° 下对应基座 X 轴（或某非 Z 方向），
不是基座 Z 轴。所以变换后 `rz_base ≈ 0`，旋转完全无效。

**数值对比**（yaw=-90°）：

```
修复前 (Bug): pad [0,0,0, 0,0,0.1] → T@ → rot_base [0.1, ~0, ~0]
             rz_base = 0.0001 → 基座 Z 旋转 ≈ 0  ❌

修复后 (Fix): pad [0,0,0, 0,0.1,0] → T@ → rot_base [~0, ~0, -0.1]
             rz_base = -0.1   → 基座 Z 旋转有值  ✅
```

#### Bug 也存在于 SpaceMouse 中

由于 RelativeFrame 把旋转分量放到了错误的槽位，SpaceMouse 的 pitch 输入也无法产生基座 Z 旋转。
但实际上 `SpacemouseIntervention` 的代码正确地提取了 `expert_a[4]`（EE ry），
只是注释误写为 `expert_a[5]`（EE rz）。注释已修正。

### 13.6 修复方案

**核心修复**（`relative_env.py`）：

```python
# 修复前
action_6d[5] = action[3]  # drz → EE rz (Bug!)
# rx=0, ry=0

# 修复后
action_6d[4] = action[3]  # drz → EE ry (EE Y ≡ 基座 -Z, 正确映射)
# rx=0, rz=0
```

**注释修正**（`wrappers.py`）：

```python
# 修复前
# 只保留 dx,dy,dz,drz: expert_a[0:3] + expert_a[5] + gripper  ← 注释与代码不一致

# 修复后
# 只保留 dx,dy,dz,dry: expert_a[0:3] + expert_a[4] (EE ry→基座Z旋转)  ← 注释与代码一致
```

**MarvinEnv** 无需修改 — `action[3]` 已经是修好后的 `rz_base`，直接用于 `action_rot_rad[2]`。

### 13.7 数学等价性：6D+mask ≡ 5D fixed

**6D+mask 路径**：
```
rz_base = (R @ [rx_ee, ry_ee, rz_ee])[2]   # 全变换后取 Z 分量
        = R[2,0]·rx_ee + R[2,1]·ry_ee + R[2,2]·rz_ee
mask 后 rx_base=0, ry_base=0:
        → rz_base = R[2,1]·ry_ee  (mask 后只剩这一项有效)
```

**5D fixed 路径**：
```
pad [0, ry_ee, 0] → T @ → rot_base = ry_ee · R[:,1]
齐次: rz_base = R[2,1]·ry_ee
```

两路径的 `rz_base = R[2,1]·ry_ee`，而 `R[2,1] = -1.0`（在 roll=-90°, pitch=0° 下），
所以 `rz_base = -ry_ee`。**完全等价**。

**重要**：5D fixed 路径反而更干净 — 它预先只把 ry 填入 6D 向量，rx/rz 完全为 0，
不存在因 pitch 非精确零度导致的 rx/rz 微量泄漏。这是相对于 6D+mask 路径的数学优势。

### 13.8 验证结果

使用 `test_action_pipeline.py` 验证了 3 个 yaw 角 (-90°, 0°, 90°) × 12 个测试用例
= **36 个场景全部通过**。位置精确一致（误差 < 1e-9 mm），旋转差异 < 1e-4 rad (~0.006°)。

详见测试脚本：`test_action_pipeline.py`

### 13.9 关键发现总结

1. **EE Y 轴恒等于基座 -Z** — 这是 roll=-90°, pitch=0° 姿态下与 yaw 无关的几何不变量
2. **5D 降维必须把旋转填入 EE ry 槽位** — 因为只有 EE ry 能映射为基座 Z 旋转
3. **5D fixed 比 6D+mask 更干净** — 避免了非零 pitch 导致的 rx/rz 微量泄漏
4. **SpaceMouse 代码的正确性** — `expert_a[4]` (EE ry) 是提取旋转的正确分量
5. **方案 A 适用于 yaw=0° 或 90°** — 因为 EE Y ≡ 基座 -Z 与 yaw 无关

---

**文档版本**: 3.0
**最后更新**: 2026-07-27
