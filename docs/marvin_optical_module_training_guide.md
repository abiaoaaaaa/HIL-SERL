# Marvin 光模块插拔任务训练指南

本文档详细介绍如何使用 HIL-SERL 框架在 Marvin 机械臂上训练光模块插拔任务。

## 任务概述

**任务名称**: 光模块插拔 (Optical Module Insertion)  
**机器人**: Marvin 六轴机械臂  
**任务类型**: 精密插入任务，类似 USB 插拔  
**难度**: 中等（需要精确定位和力控）

## 1. 环境准备

### 1.1 硬件要求

- **机械臂**: Marvin M6-40
- **夹爪**: CAN 总线夹爪（Motor ID: 1）
- **相机**: 3 个 RealSense 相机
  - 2 个腕部相机 (`wrist_1`, `wrist_2`)
  - 1 个侧面相机 (`side_policy` 和 `side_classifier` 共享)
- **人机交互**: SpaceMouse + 脚踏板

### 1.2 软件环境

确保已完成以下安装：

```bash
# 1. 激活 Python 环境
conda activate serl

# 2. 检查 Marvin SDK 是否正确安装
ls serl_robot_infra/marvin_env/SDK_PYTHON/

# 3. 检查运动学配置文件
ls serl_robot_infra/marvin_env/SDK_PYTHON/ccs_m6_40.MvKDCfg
```

### 1.3 机器人连接

启动 Marvin 机器人（无需像 Franka 那样启动独立的 server）：

```bash
# 确认机器人 IP
ping 192.168.14.190

# 检查网络连接
# Marvin 使用 UDP 直连，无需启动额外的 server 进程
```

**注意**: Marvin 环境直接通过 SDK 连接机器人，不需要像 Franka 那样运行独立的 `launch_right_server.sh`。

## 2. 任务配置

### 2.1 配置文件位置

任务配置文件位于：
```
examples/experiments/marvin_usb_insertion/config.py
```

### 2.2 关键配置项

#### 机器人连接
```python
ROBOT_IP = "192.168.14.190"  # 机器人 IP 地址
ARM = 'A'  # 使用 A 臂
KINE_CONFIG_PATH = "/path/to/ccs_m6_40.MvKDCfg"  # 运动学配置文件路径
```

#### 相机配置
```python
REALSENSE_CAMERAS = {
    "wrist_1": {"serial_number": "427622274205", ...},
    "wrist_2": {"serial_number": "427622272953", ...},
    "side_policy": {"serial_number": "036422060870", ...},
    "side_classifier": {"serial_number": "036422060870", ...},  # 与 side_policy 共享
}
```

**重要**: 使用 RealSense Viewer 查看你的相机序列号：
```bash
realsense-viewer
```

#### 图像裁剪
```python
IMAGE_CROP = {
    "wrist_1": lambda img: img[185:-41, 183:-142],
    "wrist_2": lambda img: img[92:-60, 231:-60],
    "side_policy": lambda img: img[242:-105, 495:-110],
    "side_classifier": lambda img: img[362:-251, 493:-445],
}
```

使用可视化工具调整裁剪参数（见步骤 3.1）。

#### 任务位姿

```python
# 重置位置（episode 开始位置，单位：毫米和度）
RESET_POSE = np.array([394.3, 321.7, 200.3, -90.0, 0.001, -90.001])

# 安全边界（防止机器人碰撞）
ABS_POSE_LIMIT_LOW = np.array([325.8, 274.1, 123.1, -111.9, -11.9, -99.0])
ABS_POSE_LIMIT_HIGH = np.array([518.6, 400.3, 405.7, -79.2, 12.8, -76.6])
```

**如何获取当前位姿**:
```python
# 在 Python 中运行
from marvin_env.envs.marvin_env import MarvinEnv
env = MarvinEnv()
current_pose = env.curr_pos  # [x, y, z, A, B, C] 单位：mm 和度
print(f"当前位姿: {current_pose}")
env.close()
```

#### 控制参数

```python
# 控制频率
HZ = 10  # 10Hz 控制（可提高到 20Hz）

# 动作缩放
ACTION_SCALE = np.array([20.0, 0.05, 1.0])  # [位置mm, 旋转rad, 夹爪]

# 阻抗控制模式
IMPEDANCE_MODE = "cartesian"  # "cartesian" 或 "joint"

# 柔顺参数（用于 RL 训练）
COMPLIANCE_PARAM = {
    "K": np.array([6000.0, 6000.0, 6000.0, 600.0, 600.0, 600.0, 20.0]),
    "D": np.array([0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 1.0]),
}
```

## 3. 训练流程

### 3.1 步骤 1: 调整相机裁剪和曝光

运行可视化脚本查看相机视图：

```bash
cd examples
python record_success_fail.py --exp_name marvin_usb_insertion --dry_run
```

在显示的图像中检查：
- ✅ 光模块和插座是否在视野内
- ✅ 图像是否过曝/欠曝
- ✅ 裁剪区域是否合适

根据需要调整 `config.py` 中的 `IMAGE_CROP` 和 `exposure` 参数。

### 3.2 步骤 2: 采集奖励分类器数据

奖励分类器用于判断插入是否成功。

```bash
cd examples
python record_success_fail.py --exp_name marvin_usb_insertion --successes_needed 200
```

**操作说明**:
- 用 SpaceMouse 控制机器人移动光模块
- **按住空格键**：标记当前帧为"成功"（光模块完全插入）
- **不按空格**：标记为"失败"（光模块未插入/部分插入/错误位置）

**采集策略**:
- 成功样本（~200）：光模块完全插入插座
- 失败样本（~400-600）：
  - 光模块在空中各个位置
  - 光模块接近但未插入插座
  - 光模块部分插入
  - 光模块插入错误位置

**TIP**: 为了训练鲁棒的分类器，失败样本应该是成功样本的 2-3 倍。

数据保存位置：`examples/experiments/marvin_usb_insertion/classifier_data/`

### 3.3 步骤 3: 训练奖励分类器

```bash
cd examples/experiments/marvin_usb_insertion
python ../../train_reward_classifier.py --exp_name marvin_usb_insertion
```

训练完成后，分类器保存在：`examples/experiments/marvin_usb_insertion/classifier_ckpt/`

**验证分类器**:
```bash
# 重新运行数据采集脚本，观察分类器输出
python ../../examples/record_success_fail.py --exp_name marvin_usb_insertion --verify_classifier
```

### 3.4 步骤 4: 录制示范数据

录制 20 条成功的插入示范：

```bash
cd examples
python record_demos.py --exp_name marvin_usb_insertion --successes_needed 20
```

**操作说明**:
- 使用 SpaceMouse 演示完整的插入过程
- 每次成功插入后，机器人会自动重置
- 脚本会自动保存成功的轨迹

**TIP**: 
- 尽量多样化示范轨迹（不同的抓取位置、插入角度）
- 确保示范动作流畅、自然
- 如果分类器误判（假阳性/假阴性），返回步骤 3.2 补充数据

数据保存位置：`examples/experiments/marvin_usb_insertion/demo_data/`

### 3.5 步骤 5: 强化学习训练

训练分为两个进程：**Actor**（采集数据）和 **Learner**（训练策略）。

#### 5.1 启动 Actor

```bash
cd examples/experiments/marvin_usb_insertion
bash run_actor.sh
```

`run_actor.sh` 内容示例：
```bash
#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python ../../async_drq_randomized.py \
    --exp_name=marvin_optical_module \
    --checkpoint_path=/path/to/checkpoints \
    --learner_ip=localhost \
    --learner_port=5555 \
    --actor \
    --eval_checkpoint_step=0 \
    --eval_n_trajs=0
```

#### 5.2 启动 Learner

在**另一个终端**中运行：

```bash
cd examples/experiments/marvin_usb_insertion
bash run_learner.sh
```

`run_learner.sh` 内容示例：
```bash
#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python ../../async_drq_randomized.py \
    --exp_name=marvin_optical_module \
    --checkpoint_path=/path/to/checkpoints \
    --demo_path=/path/to/demo_data/demo_*.pkl \
    --learner \
    --learner_port=5555 \
    --batch_size=256 \
    --max_steps=1000000
```

**TIP**: 如果要恢复训练，只需将 `checkpoint_path` 指向之前的保存目录，代码会自动加载最新的 checkpoint。

### 3.6 步骤 6: 人工干预 (Human Intervention)

训练过程中，你需要通过 SpaceMouse 给予人工干预以加速学习：

#### 干预策略

**训练初期（前 30-60 分钟）**:
- 频率：每 1-2 个 episode 干预一次
- 目的：引导策略探索正确的行为
- 示例：
  - 策略随机移动 20-30 步后，介入将光模块移到插座附近
  - 让策略尝试插入动作
  - 如果策略持续做错误动作（如远离插座），立即干预纠正

**训练中期（策略开始成功）**:
- 频率：降低到每 5-10 个 episode 干预一次
- 目的：纠正重复的错误行为
- 示例：
  - 策略反复卡在某个位置，干预帮助通过
  - 策略插入角度错误，干预调整角度

**训练后期（接近收敛）**:
- 频率：几乎不干预
- 目的：让策略自主完成，仅在极端情况下干预
- 示例：
  - 练习边缘情况的恢复（如掉落后重新抓取）

#### 干预技巧

1. **平衡探索与干预**: 让策略有足够的探索空间，但不浪费时间在明显错误的行为上

2. **引导而非代劳**: 将策略引导到关键位置（如插座附近），然后让它自己完成插入

3. **频繁获得奖励**: 训练初期，让策略较频繁地获得成功奖励（~1/3 的 episode），帮助 value 函数快速收敛

4. **练习恢复行为**: 后期可以故意制造失败场景（如中途松开光模块），让策略学会恢复

#### 控制键位

- **SpaceMouse 控制**: 末端移动和旋转
- **F1 键**: 切换夹爪开/关
- **脚踏板（Shift + ←）**: 手动标记成功（用于分类器训练不准的情况）

### 3.7 步骤 7: 监控训练

训练过程中关注以下指标：

**终端输出**:
```
[step=1000] reward=0.05 success_rate=0.10 actor_loss=0.5 critic_loss=1.2
[step=2000] reward=0.15 success_rate=0.25 actor_loss=0.3 critic_loss=0.9
...
[step=50000] reward=0.85 success_rate=0.95 actor_loss=0.1 critic_loss=0.3
```

**收敛标志**:
- ✅ Success rate > 95%
- ✅ 策略能稳定完成插入
- ✅ Actor/Critic loss 收敛

**训练时间参考**:
- 预期训练时间：**2-3 小时**（取决于任务难度和干预质量）
- Checkpoint 保存：每 2000 步
- Buffer 保存：每 1000 步

## 4. 策略评估

训练完成后，评估策略性能：

```bash
cd examples/experiments/marvin_usb_insertion
# 修改 run_actor.sh，添加评估参数：
# --eval_checkpoint_step=50000
# --eval_n_trajs=20

bash run_actor.sh
```

评估会运行 20 次完整的插入任务，输出成功率。

## 5. 故障排查

### 5.1 机器人连接问题

**问题**: 无法连接到 Marvin 机器人

```bash
# 检查网络
ping 192.168.14.190

# 检查防火墙
sudo ufw status

# 检查端口（Marvin 使用 UDP 8000）
sudo netstat -tulpn | grep 8000
```

参考：`~/.claude/projects/-home-xlb-code-marvin-hil-serl/memory/marvin-troubleshooting.md`

### 5.2 相机问题

**问题**: 相机初始化失败

```bash
# 检查相机连接
realsense-viewer

# 检查 USB 权限
sudo usermod -a -G video $USER  # 需要重新登录
```

### 5.3 分类器问题

**问题**: 分类器误判（假阳性/假阴性）

**解决方案**:
1. 返回步骤 3.2，针对误判场景补充数据
2. 调整分类器阈值（`config.py` 中的 `reward_func`）
3. 增加分类器相机数量或调整裁剪区域

### 5.4 训练不收敛

**可能原因**:
- 示范数据质量差 → 重新录制
- 人工干预过多/过少 → 调整干预策略
- 奖励稀疏 → 检查分类器是否正常工作
- 控制参数不合适 → 调整 `COMPLIANCE_PARAM` 或 `ACTION_SCALE`

## 6. 高级配置

### 6.1 调整控制频率

```python
# config.py
HZ = 20  # 从 10Hz 提高到 20Hz，响应更快
```

**注意**: 更高频率需要更强的计算能力和更低的网络延迟。

### 6.2 切换阻抗控制模式

```python
# config.py
IMPEDANCE_MODE = "joint"  # 切换到关节阻抗（更柔顺）
```

**使用场景**:
- `"cartesian"`: 精确定位，适合插入任务（默认）
- `"joint"`: 柔顺交互，适合力控任务

### 6.3 调整末端姿态控制

```python
# config.py
FIXED_ORIENTATION = True  # 锁定末端姿态
FIXED_ORIENTATION_ABC = [-90.0, 0.0, -90.0]  # [A, B, C] 度
```

**使用场景**: 如果任务只需要平移（不需要旋转），可以锁定姿态简化策略学习。

### 6.4 随机重置

```python
# config.py
RANDOM_RESET = True  # 启用随机重置
RANDOM_XY_RANGE = 8.0  # XY 平面随机 ±8mm
RANDOM_RZ_RANGE = 0.08  # 绕 Z 轴随机 ±0.08rad
```

**作用**: 增加初始位置的多样性，提升策略鲁棒性。

## 7. 与 Franka 的主要差异

| 特性 | Franka Arm | Marvin Arm |
|------|------------|------------|
| **通信方式** | HTTP (ROS) | UDP 直连 |
| **Server 进程** | 需要 `launch_right_server.sh` | 不需要 |
| **控制频率** | 通常 10Hz | 10-20Hz（可达 1KHz） |
| **力/力矩反馈** | 需要外部传感器 | 原生支持 |
| **位姿单位** | 米 + 四元数 | 毫米 + 欧拉角 |
| **IK 求解** | 外部 | SDK 内置 |
| **轨迹规划** | 需要自行实现 | MovL/MovLA 内置 |

## 8. 总结

### 完整训练流程

1. ✅ 调整相机裁剪和曝光
2. ✅ 采集奖励分类器数据（~600 样本，2:1 失败:成功）
3. ✅ 训练奖励分类器
4. ✅ 录制示范数据（20 条轨迹）
5. ✅ 启动 RL 训练（Actor + Learner）
6. ✅ 给予人工干预（初期频繁，后期减少）
7. ✅ 监控训练，等待收敛（2-3 小时）
8. ✅ 评估策略性能

### 关键成功因素

- **高质量的分类器数据**: 充分覆盖失败模式
- **多样化的示范轨迹**: 不同的抓取和插入方式
- **适当的人工干预**: 引导探索，不过度代劳
- **合理的控制参数**: 柔顺度、动作缩放、安全边界

### 预期结果

- **成功率**: > 95%
- **训练时间**: 2-3 小时
- **数据量**: ~50-100k transitions

## 附录

### A. 常用命令

```bash
# 激活环境
conda activate serl

# 查看相机
realsense-viewer

# 获取当前位姿
python -c "from marvin_env.envs.marvin_env import MarvinEnv; env = MarvinEnv(); print(env.curr_pos); env.close()"

# 测试夹爪
python serl_robot_infra/marvin_env/SDK_PYTHON/test_gripper.py

# 查看训练进度
tail -f /path/to/checkpoints/training.log
```

### B. 配置文件模板

参考：`examples/experiments/marvin_usb_insertion/config.py`

### C. 相关文档

- Marvin SDK 文档：`.marvin参考/TJ_FX_ROBOT_CONTRL_SDK-master/`
- HIL-SERL 算法文档：`HIL_SERL_算法完整技术文档.md`
- Marvin 环境集成：`serl_robot_infra/marvin_env/INTEGRATION_SUMMARY.md`
