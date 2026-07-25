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

## 注意事项

1. 所有工具的输出文件都保存在 `utils/debug_tools/` 目录下
2. 运行前确保机器人已连接并且在安全状态
3. 使用 Ctrl+C 可以随时中断程序
