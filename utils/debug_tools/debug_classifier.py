#!/usr/bin/env python3
"""
调试奖励分类器 - Debug Reward Classifier

功能：
    实时读取相机图像并使用训练好的分类器判断任务成功/失败状态。
    主要用于调试分类器的准确性和阈值设置。

主要功能：
    1. 实时显示分类器输出（logit 原始值和 sigmoid 概率值）
    2. 自动保存每帧图像到对应的成功/失败文件夹
    3. 显示分类统计信息（成功率、失败率、总帧数）
    4. 帮助调试分类器阈值和训练效果

工作流程：
    1. 初始化：
       - 加载训练好的分类器模型（checkpoint）
       - 初始化 RealSense 相机
       - 创建输出目录（success/failure）

    2. 主循环（实时分类）：
       - 从相机读取一帧图像
       - 预处理图像（裁剪、归一化）
       - 输入分类器获取预测结果
       - 显示结果（logit、sigmoid、分类结果）
       - 保存图像到对应文件夹
       - 更新统计信息

    3. 退出：
       - Ctrl+C 中断
       - 显示最终统计信息
       - 释放相机资源

配置说明：
    - CLASSIFIER_CHECKPOINT: 分类器模型检查点路径
    - CLASSIFIER_KEY: 使用哪个相机的分类器（如 "side_classifier"）
    - CAMERA_CONFIG: 相机参数（序列号、分辨率、帧率、曝光）
    - IMAGE_CROP: 图像裁剪函数（与训练时保持一致）
    - OUTPUT_DIR: 输出目录（保存分类结果图像）

输出：
    - 控制台：实时显示分类结果和统计信息
    - 文件：图像保存到 utils/debug_tools/debug_classifier_output/
           ├── success/  - 分类为成功的图像
           └── failure/  - 分类为失败的图像

使用方法：
    cd /home/xlb/code_marvin/hil-serl
    python utils/debug_tools/debug_classifier.py

    按 Ctrl+C 停止

注意事项：
    1. 确保相机已连接且序列号正确
    2. 确保分类器检查点文件存在
    3. 图像裁剪参数必须与训练时一致
    4. 输出目录会占用磁盘空间，注意定期清理
"""
import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from serl_launcher.networks.reward_classifier import load_classifier_func
from franka_env.camera.rs_capture import RSCapture

# ==================== 配置 ====================
CLASSIFIER_CHECKPOINT = "/home/xlb/code_marvin/hil-serl/examples/classifier_ckpt/checkpoint_150"
CLASSIFIER_KEY = "side_classifier"

# 相机配置（与config.py保持一致）
CAMERA_CONFIG = {
    "serial_number": "036422060870",
    "dim": (1280, 720),
    "fps": 30,
    "exposure": 13000,
}

# 图像裁剪（与config.py保持一致）
IMAGE_CROP = lambda img: img[163:-4, 428:-3]

# 输出目录
OUTPUT_DIR = Path("/home/xlb/code_marvin/hil-serl/utils/debug_tools/debug_classifier_output")
OUTPUT_DIR.mkdir(exist_ok=True)

SUCCESS_DIR = OUTPUT_DIR / "success"
FAILURE_DIR = OUTPUT_DIR / "failure"
SUCCESS_DIR.mkdir(exist_ok=True)
FAILURE_DIR.mkdir(exist_ok=True)

# 分类器阈值
SUCCESS_THRESHOLD = 0.7

# ==================== 初始化 ====================
print("[1/3] 初始化相机...")
cap = RSCapture(
    name="side_classifier",
    serial_number=CAMERA_CONFIG["serial_number"],
    dim=CAMERA_CONFIG["dim"],
    fps=CAMERA_CONFIG["fps"],
    exposure=CAMERA_CONFIG["exposure"]
)
# RSCapture在__init__中已经自动启动，不需要调用start()

print("[2/3] 加载分类器...")
print(f"  - Checkpoint: {CLASSIFIER_CHECKPOINT}")
print(f"  - Image key: {CLASSIFIER_KEY}")

# 先读取一帧真实图像
print("  - 读取测试图像...")
success, test_image = cap.read()
if not success or test_image is None:
    print("[ERROR] 无法读取测试图像")
    exit(1)

# 裁剪并resize
cropped = IMAGE_CROP(test_image)
resized = cv2.resize(cropped, (128, 128))
print(f"  - 图像尺寸: {resized.shape}, dtype: {resized.dtype}")

# 创建dummy observation
dummy_obs = {CLASSIFIER_KEY: resized[None, ...]}  # [1, 128, 128, 3]

print("  - 加载分类器模型...")
try:
    classifier_func = load_classifier_func(
        key=jax.random.PRNGKey(0),
        sample=dummy_obs,
        image_keys=[CLASSIFIER_KEY],
        checkpoint_path=CLASSIFIER_CHECKPOINT,
    )
    print("  ✅ 分类器加载成功")
except Exception as e:
    print(f"[ERROR] 分类器加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("[3/3] 开始采集...")
print(f"成功图像保存到: {SUCCESS_DIR}")
print(f"失败图像保存到: {FAILURE_DIR}")
print(f"成功阈值: sigmoid > {SUCCESS_THRESHOLD}")
print("-" * 80)

# ==================== 主循环 ====================
frame_count = 0
success_count = 0
failure_count = 0

try:
    while True:
        # 读取图像
        success, raw_image = cap.read()
        if not success or raw_image is None:
            print("[ERROR] 无法读取图像")
            time.sleep(0.1)
            continue

        # 裁剪
        cropped = IMAGE_CROP(raw_image)

        # Resize到128x128（与训练时一致）
        resized = cv2.resize(cropped, (128, 128))

        # 准备observation
        obs = {CLASSIFIER_KEY: resized[None, ...]}  # [1, 128, 128, 3]

        # 运行分类器
        classifier_logit = classifier_func(obs)
        logit_val = float(classifier_logit.item())

        # 计算sigmoid
        sigmoid_val = 1.0 / (1.0 + np.exp(-logit_val))

        # 判断成功/失败
        is_success = sigmoid_val > SUCCESS_THRESHOLD

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if is_success:
            save_dir = SUCCESS_DIR
            success_count += 1
            label = "SUCCESS"
        else:
            save_dir = FAILURE_DIR
            failure_count += 1
            label = "FAILURE"

        filename = f"{timestamp}_logit{logit_val:.3f}_sigmoid{sigmoid_val:.3f}.jpg"
        save_path = save_dir / filename

        # 在图像上绘制信息
        display_img = resized.copy()
        color = (0, 255, 0) if is_success else (0, 0, 255)
        cv2.putText(display_img, f"{label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_img, f"Logit: {logit_val:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f"Sigmoid: {sigmoid_val:.3f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(str(save_path), display_img)

        # 打印信息
        frame_count += 1
        print(f"[Frame {frame_count:04d}] "
              f"Logit={logit_val:+7.3f} | Sigmoid={sigmoid_val:.3f} | "
              f"{label:7s} | Success={success_count:3d} Failure={failure_count:3d}")

        # 控制采样频率（每秒1帧）
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n" + "=" * 80)
    print("🛑 用户中断")
    print(f"总帧数: {frame_count}")
    print(f"成功: {success_count} ({success_count/max(frame_count,1)*100:.1f}%)")
    print(f"失败: {failure_count} ({failure_count/max(frame_count,1)*100:.1f}%)")
    print(f"图像保存在: {OUTPUT_DIR}")

finally:
    cap.stop()
    print("✅ 相机已关闭")
