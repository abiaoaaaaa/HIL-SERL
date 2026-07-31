# Debug Tools

调试工具集合，用于诊断和分析机器人系统的运行状态。

## 文件说明

### debug_classifier.py
**用途**: 调试奖励分类器
- 实时显示分类器输出（logit和sigmoid值）
- 保存每帧图像到文件夹（成功/失败分类）
- 帮助调试分类器阈值和训练效果

**输出目录**: `utils/debug_tools/debug_classifier_output/`

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/debug_tools/debug_classifier.py
```

### debug_movla_spacemouse.py
**用途**: 模拟Space Mouse控制测试
- 模拟执行控制周期
- 记录规划的轨迹点和实际执行位置
- 数据保存到JSON文件用于分析

**输出**: JSON文件保存在 `utils/debug_tools/` 目录

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/debug_tools/debug_movla_spacemouse.py --hz 10 --simulate   # 模拟 10Hz
python utils/debug_tools/debug_movla_spacemouse.py --hz 20 --simulate   # 模拟 20Hz
```

### debug_step_timing.py
**用途**: 完全模拟 marvin_env.py step() 的数据记录
- 记录每个step的详细时序信息
- 分析movLA规划、发送轨迹、sleep等各阶段耗时
- 帮助优化控制频率和性能

**输出**: JSON文件保存在 `utils/debug_tools/` 目录

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/debug_tools/debug_step_timing.py --hz 10
python utils/debug_tools/debug_step_timing.py --hz 20
```

### replay_marvin_action_trajectory.py
**用途**: 不使用相机或观测，回放 buffer 中的一条 Marvin 动作轨迹，复现动作缩放、坐标变换、EMA、10 Hz 到两个 50 ms 子步的执行过程。

**movLA 时间语义（2026-07-30 记录）**:

- `movLA` 返回的点集不是单纯的空间折线；它由 `vel`、`acc` 和 `freq_hz` 共同定义，点数对应规划轨迹的时间采样。
- 不能只为满足控制周期，把 `N` 个规划点重采样为 10 个点后固定在 50 ms 内发送。这会把规划轨迹压缩；例如按 200 Hz 规划的 106 点约对应 530 ms，若 50 ms 播放就是约 10.6 倍加速，规划的速度/加速度约束不再成立，可能导致抖动。
- 回放日志的 `movla_execution` 会记录原始点数、规划频率、推定规划时长、实际播放时长和 `trajectory_compression_ratio`。只要该比值大于 1，就应把本次结果视为“时间压缩实验”，而不是遵守 movLA 约束的回放。
- 在确认 SDK 返回点的实际时间基准前，不应把 200 Hz 的均匀 `send_cmd()` 误解为 movLA 的正确执行方式；它只保证发送节奏，并不保证规划速度/加速度得到保持。

## 注意事项

1. 所有工具的输出文件都保存在 `utils/debug_tools/` 目录下
2. 运行前确保机器人已连接并且在安全状态
3. 使用 Ctrl+C 可以随时中断程序
