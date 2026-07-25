# Visualization Tools

可视化和分析工具集合，用于查看相机数据、分析轨迹数据和生成图表。

## 文件说明

### analyze_trajectory_data.py
**用途**: 深度分析轨迹数据JSON文件
- 分析单个轨迹文件或对比两个文件
- 时间分布分析（movLA规划、发送、sleep时间）
- 轨迹点数量统计
- 关节角度变化分析（单步内、步间、连续性）
- 规划轨迹 vs 实际执行对比
- 异常模式检测（回跳、震荡）

**输入**: 由 debug_movla_spacemouse.py 或 debug_step_timing.py 生成的 JSON 文件

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl

# 分析单个文件
python utils/visualization_tools/analyze_trajectory_data.py \
    utils/debug_tools/trajectory_data_10hz_20240125.json

# 对比两个文件（10Hz vs 20Hz）
python utils/visualization_tools/analyze_trajectory_data.py \
    utils/debug_tools/trajectory_data_10hz_20240125.json \
    utils/debug_tools/trajectory_data_20hz_20240125.json
```

**输出内容**:
- 时间分布统计（平均、最大值）
- 轨迹点数量统计
- 关节角度变化详细分析
- 异常模式检测报告（回跳率、震荡）
- 两文件对比表格

### view_cameras.py
**用途**: 相机快照工具
- 从所有配置的相机读取一帧
- 保存到文件方便后续分析
- 显示相机状态和图像信息
- 用于验证相机连接和配置

**输出**: `utils/visualization_tools/camera_snapshots/`

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/visualization_tools/view_cameras.py
```

**输出文件**: 
- `camera_snapshots/<camera_name>_<timestamp>.jpg` - 每个相机的快照

**注意事项**:
- 会读取 marvin_usb_insertion/config.py 中的相机配置
- 确保相机已连接且序列号正确
- 使用 `rs-enumerate-devices` 查看可用相机

### visualize_trajectory.py
**用途**: 轨迹数据可视化生成图表
- 对比不同控制频率的轨迹（如10Hz vs 20Hz）
- 生成关节角度轨迹图
- 生成时间分析图（movLA、发送、总时间）
- 生成发送时间戳分析图（检测抖动）

**输入**: 两个不同频率的轨迹数据JSON文件

**输出**: PNG图表文件

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl

python utils/visualization_tools/visualize_trajectory.py \
    utils/debug_tools/trajectory_data_10hz_20240125.json \
    utils/debug_tools/trajectory_data_20hz_20240125.json \
    --output ./trajectory_analysis
```

**生成的图表**:
1. `joint_trajectories_comparison.png` - 6个关节的角度轨迹对比（10Hz vs 20Hz）
2. `timing_analysis.png` - 时间分析4图：
   - movLA规划时间
   - 轨迹点发送时间
   - Step总执行时间 vs 控制周期
   - 规划点数变化
3. `send_timestamps.png` - 每个点的发送耗时分析（检测抖动）

**参数说明**:
- `file_10hz`: 10Hz控制频率的数据文件（位置参数1）
- `file_20hz`: 20Hz控制频率的数据文件（位置参数2）
- `--output`: 输出目录（默认: ./trajectory_analysis）

## 工作流程示例

### 完整的轨迹分析流程

1. **记录数据** (使用 debug_tools)
```bash
# 记录10Hz数据
python utils/debug_tools/debug_step_timing.py --hz 10

# 记录20Hz数据
python utils/debug_tools/debug_step_timing.py --hz 20
```

2. **文本分析** (快速查看统计信息)
```bash
python utils/visualization_tools/analyze_trajectory_data.py \
    utils/debug_tools/trajectory_data_10hz_*.json \
    utils/debug_tools/trajectory_data_20hz_*.json
```

3. **可视化分析** (生成图表深入分析)
```bash
python utils/visualization_tools/visualize_trajectory.py \
    utils/debug_tools/trajectory_data_10hz_*.json \
    utils/debug_tools/trajectory_data_20hz_*.json \
    --output ./trajectory_analysis
```

4. **查看结果**
   - 文本统计: 控制台输出
   - 图表: `./trajectory_analysis/` 目录下的PNG文件

### 相机配置流程

1. **获取相机快照**
```bash
python utils/visualization_tools/view_cameras.py
```

2. **配置裁剪区域**
```bash
python utils/test_tools/test_camera_crop.py
```

## 依赖项

- **analyze_trajectory_data.py**: `json`, `numpy`
- **view_cameras.py**: `cv2`, `franka_env`
- **visualize_trajectory.py**: `json`, `numpy`, `matplotlib`

## 注意事项

1. **路径**: 所有工具从项目根目录运行
2. **输出**: 
   - `view_cameras.py` 输出到 `utils/visualization_tools/camera_snapshots/`
   - `visualize_trajectory.py` 输出目录通过 `--output` 参数指定
3. **数据来源**: 这些工具通常分析 debug_tools 生成的数据
4. **中文支持**: matplotlib图表使用中文标签，可能需要配置中文字体

## 迁移记录

这些文件从以下位置迁移而来：
- `examples/experiments/marvin_usb_insertion/analyze_trajectory_data.py` → `utils/visualization_tools/`
- `examples/experiments/marvin_usb_insertion/view_cameras.py` → `utils/visualization_tools/`
- `examples/experiments/marvin_usb_insertion/visualize_trajectory.py` → `utils/visualization_tools/`

迁移时已调整：
- ✅ view_cameras.py 的项目路径引用（`sys.path`）
- ✅ view_cameras.py 的相机快照保存路径
- ✅ 其他两个文件无需路径调整（纯数据分析工具）
- ✅ 所有文件语法检查通过
