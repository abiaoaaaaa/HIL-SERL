# Utils - 工具集合

这个目录包含了项目的调试工具和测试工具。

## 目录结构

```
utils/
├── debug_tools/          # 调试工具
│   ├── debug_classifier.py
│   ├── debug_movla_spacemouse.py
│   ├── debug_step_timing.py
│   └── README.md
├── test_tools/           # 测试工具
│   ├── test_camera_crop.py
│   ├── test_classifier_accuracy.py
│   ├── test_marvin_env.py
│   ├── test_simple_movement.py
│   ├── test_spacemouse_control.py
│   └── README.md
├── visualization_tools/  # 可视化工具
│   ├── analyze_trajectory_data.py
│   ├── view_cameras.py
│   ├── visualize_trajectory.py
│   └── README.md
├── analyze_policy_jitter.py
├── drag_and_measure.py
├── get_camera_serials.py
├── inspect_buffer.py       # Buffer数据检查工具
├── BUFFER_FORMAT.md        # Buffer格式文档
└── test_deep.py
```

## 工具分类

### Debug Tools (调试工具)
用于诊断和分析系统运行状态：
- **debug_classifier.py** - 调试奖励分类器，实时显示分类结果
- **debug_movla_spacemouse.py** - Space Mouse控制数据记录和分析
- **debug_step_timing.py** - step()执行时序分析

详细说明请查看 [debug_tools/README.md](debug_tools/README.md)

### Test Tools (测试工具)
用于验证和配置系统组件：
- **test_camera_crop.py** - 相机裁剪区域配置
- **test_classifier_accuracy.py** - 分类器准确率测试
- **test_marvin_env.py** - Marvin环境基础功能测试
- **test_simple_movement.py** - 简单移动测试
- **test_spacemouse_control.py** - Space Mouse控制测试

详细说明请查看 [test_tools/README.md](test_tools/README.md)

### Visualization Tools (可视化工具)
用于数据分析和可视化：
- **analyze_trajectory_data.py** - 深度分析轨迹数据，检测异常模式
- **view_cameras.py** - 相机快照工具，验证相机配置
- **visualize_trajectory.py** - 生成轨迹对比图表

详细说明请查看 [visualization_tools/README.md](visualization_tools/README.md)

### Buffer Analysis Tools (Buffer分析工具)
用于分析replay buffer数据：
- **analyze_policy_jitter.py** - 分析Policy和人工演示的动作抖动、episode统计
- **inspect_buffer.py** - 快速检查buffer文件格式和内容
- **BUFFER_FORMAT.md** - Buffer数据格式详细文档

### Other Tools (其他工具)
- **drag_and_measure.py** - 拖拽测量工具
- **get_camera_serials.py** - 获取相机序列号

## 使用说明

所有工具都需要从项目根目录运行：

```bash
cd /home/xlb/code_marvin/hil-serl

# 运行debug工具
python utils/debug_tools/debug_classifier.py
python utils/debug_tools/debug_step_timing.py --hz 10

# 运行test工具
python utils/test_tools/test_camera_crop.py
python utils/test_tools/test_simple_movement.py

# 运行可视化工具
python utils/visualization_tools/view_cameras.py
python utils/visualization_tools/analyze_trajectory_data.py data.json
python utils/visualization_tools/visualize_trajectory.py data_10hz.json data_20hz.json

# 分析buffer数据
python utils/analyze_policy_jitter.py
python utils/analyze_policy_jitter.py --last-n-files 5
python utils/inspect_buffer.py
python utils/inspect_buffer.py --save-images --show-episodes
```

## 典型工作流程

### 轨迹分析工作流
```bash
# 1. 记录数据
python utils/debug_tools/debug_step_timing.py --hz 10
python utils/debug_tools/debug_step_timing.py --hz 20

# 2. 文本分析
python utils/visualization_tools/analyze_trajectory_data.py \
    utils/debug_tools/trajectory_data_10hz_*.json \
    utils/debug_tools/trajectory_data_20hz_*.json

# 3. 可视化
python utils/visualization_tools/visualize_trajectory.py \
    utils/debug_tools/trajectory_data_10hz_*.json \
    utils/debug_tools/trajectory_data_20hz_*.json \
    --output ./trajectory_analysis
```

### 相机配置工作流
```bash
# 1. 获取相机快照
python utils/visualization_tools/view_cameras.py

# 2. 配置裁剪区域
python utils/test_tools/test_camera_crop.py
```

### Buffer数据分析工作流
```bash
# 1. 快速检查buffer格式和内容
python utils/inspect_buffer.py

# 2. 保存样本图像
python utils/inspect_buffer.py --save-images --num-images 10

# 3. 深度分析动作抖动和episode统计
python utils/analyze_policy_jitter.py

# 4. 只分析最近5个文件（快速检查）
python utils/analyze_policy_jitter.py --last-n-files 5

# 5. 查看详细格式文档
cat utils/BUFFER_FORMAT.md
```

## 注意事项

1. **路径引用**: 所有工具已经调整路径，相对于项目根目录 (`/home/xlb/code_marvin/hil-serl`)
2. **输出目录**: 
   - Debug工具输出到 `utils/debug_tools/` 
   - Test工具输出到 `utils/test_tools/`
   - Visualization工具输出到 `utils/visualization_tools/camera_snapshots/` 或指定的输出目录
3. **安全性**: 运行前确保机器人处于安全状态，所有工具都支持 Ctrl+C 急停
4. **依赖**: 某些工具需要特定硬件（如Space Mouse、相机）或已训练的模型

## 迁移记录

这些文件从以下位置迁移而来：
- `examples/experiments/marvin_usb_insertion/debug_*.py` → `utils/debug_tools/`
- `examples/experiments/marvin_usb_insertion/test_*.py` → `utils/test_tools/`
- `serl_robot_infra/marvin_env/envs/test_marvin_env.py` → `utils/test_tools/`
- `examples/experiments/marvin_usb_insertion/analyze_trajectory_data.py` → `utils/visualization_tools/`
- `examples/experiments/marvin_usb_insertion/view_cameras.py` → `utils/visualization_tools/`
- `examples/experiments/marvin_usb_insertion/visualize_trajectory.py` → `utils/visualization_tools/`

迁移时已调整：
- ✅ 项目路径引用（`sys.path`）
- ✅ 输出文件保存路径
- ✅ 所有import语句
- ✅ 语法检查通过
