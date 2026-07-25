# Test Tools

测试工具集合，用于验证和配置机器人系统的各个组件。

## 文件说明

### test_camera_crop.py
**用途**: 相机裁剪区域配置工具
- 从保存的图片加载相机快照
- 鼠标拖动选择裁剪区域
- 实时预览裁剪效果
- 自动生成配置代码

**输出**: `utils/test_tools/camera_crop_config.txt`

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
# 1. 先运行 test_single_camera.py 生成相机快照
# 2. 运行本脚本
python utils/test_tools/test_camera_crop.py
```

**操作说明**:
- 鼠标拖动: 选择裁剪区域
- 按 's': 保存当前配置到文件
- 按 'r': 重置当前相机的裁剪
- 按 'p': 预览裁剪后的图像
- 按 'q': 退出

### test_classifier_accuracy.py
**用途**: 测试分类器准确率
- 从success和failure数据集加载并评估
- 计算分类器的准确率、召回率等指标
- 帮助验证分类器训练效果

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/test_tools/test_classifier_accuracy.py
```

### test_simple_movement.py
**用途**: Marvin简单移动测试
- 连接机器人
- 获取当前位姿
- 设置笛卡尔阻抗参数
- 执行简单移动（如前移30mm）
- Ctrl+C急停保护

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/test_tools/test_simple_movement.py
```

### test_spacemouse_control.py
**用途**: Space Mouse控制Marvin机械臂测试
- 通过Space Mouse (3Dconnexion)实时控制机械臂
- 支持笛卡尔阻抗和关节阻抗模式
- 测试末端执行器坐标系到基座坐标系的变换

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/test_tools/test_spacemouse_control.py
```

**操作说明**:
- 推/拉 Space Mouse: 沿末端执行器轴平移
- 倾斜/扭转: 绕末端执行器轴旋转
- 左按钮: 关闭夹爪
- 右按钮: 打开夹爪
- Ctrl+C: 急停退出

### test_marvin_env.py
**用途**: Marvin环境基础功能测试
- 环境初始化
- 观测空间验证
- step()执行测试
- reset()功能测试

**使用方法**:
```bash
cd /home/xlb/code_marvin/hil-serl
python utils/test_tools/test_marvin_env.py
```

## 注意事项

1. 运行前确保机器人已连接并且在安全状态
2. Space Mouse相关测试需要连接3Dconnexion设备
3. 相机测试需要先配置好相机序列号
4. 使用 Ctrl+C 可以随时安全中断程序
5. 所有配置文件输出到 `utils/test_tools/` 目录
