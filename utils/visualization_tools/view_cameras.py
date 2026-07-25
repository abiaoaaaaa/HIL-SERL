#!/usr/bin/env python3
"""
相机快照工具 - Camera Snapshot Tool

功能：
    从所有配置的 RealSense 相机读取一帧图像并保存为文件。
    用于验证相机连接、配置和图像质量。

主要功能：
    1. 自动读取项目配置中的所有相机
    2. 从每个相机获取一帧图像
    3. 保存图像到指定目录（带时间戳）
    4. 显示相机状态和图像信息
    5. 处理共享相机（同一物理相机多个逻辑名称）

工作流程：
    1. 初始化阶段：
       a) 读取配置：
          - 从 config.py 加载相机配置
          - 获取所有 REALSENSE_CAMERAS 定义
       b) 初始化相机：
          - 遍历所有相机配置
          - 创建 RSCapture 或 VideoCapture 实例
          - 处理共享相机（如 side_classifier 和 side_policy）
          - 显示初始化状态

    2. 图像采集阶段：
       对每个相机：
       a) 读取一帧：
          - 调用 cap.read() 获取图像
          - 验证图像有效性（非 None）
       b) 显示信息：
          - 图像尺寸 (H, W, C)
          - 数据类型（uint8）
          - 像素值范围
       c) 应用裁剪（如果配置）：
          - 从 config 读取裁剪函数
          - 应用裁剪得到处理后图像
          - 显示裁剪后尺寸

    3. 保存阶段：
       a) 创建保存目录：
          - utils/visualization_tools/camera_snapshots/
       b) 生成文件名：
          - 格式: <camera_name>_<timestamp>.jpg
          - 时间戳: YYYYMMDD_HHMMSS
       c) 保存图像：
          - 使用 cv2.imwrite() 保存为 JPEG
          - 显示保存路径

    4. 清理阶段：
       - 释放所有相机资源
       - 关闭连接

相机配置结构：
    REALSENSE_CAMERAS = {
        "camera_name": {
            "serial_number": "036422060870",  # 相机序列号
            "dim": (1280, 720),               # 分辨率
            "fps": 30,                        # 帧率
            "exposure": 13000,                # 曝光时间（微秒）
        },
        ...
    }

共享相机处理：
    某些配置中，多个逻辑名称可能对应同一物理相机：
    - 例如: "side_policy" 和 "side_classifier" 共享同一相机
    - 工具自动检测并复用已初始化的相机实例
    - 避免重复初始化导致冲突

输出目录结构：
    utils/visualization_tools/camera_snapshots/
    ├── wrist_1_20240125_143025.jpg
    ├── side_policy_20240125_143025.jpg
    ├── side_classifier_20240125_143025.jpg
    └── ...

输出示例：
    ======================================================================
    相机快照工具 - 初始化相机...
    ======================================================================

    [wrist_1]
      序列号: 130322274175
      分辨率: (1280, 720)
      曝光: 10500
    ✓ 初始化成功

    [side_policy]
      序列号: 036422060870
      分辨率: (1280, 720)
      曝光: 13000
    ✓ 初始化成功

    [side_classifier] 共享 side_policy 相机

    ======================================================================
    读取图像...
    ======================================================================

    [wrist_1]
      原始尺寸: (720, 1280, 3)
      裁剪后: (553, 849, 3)
      保存至: camera_snapshots/wrist_1_20240125_143025.jpg

    [side_policy]
      原始尺寸: (720, 1280, 3)
      裁剪后: (553, 849, 3)
      保存至: camera_snapshots/side_policy_20240125_143025.jpg

    [side_classifier]
      原始尺寸: (720, 1280, 3)
      裁剪后: (553, 849, 3)
      保存至: camera_snapshots/side_classifier_20240125_143025.jpg

    ======================================================================
    完成！所有图像已保存到 camera_snapshots/
    ======================================================================

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 运行快照工具
    python utils/visualization_tools/view_cameras.py

    # 查看保存的图像
    ls utils/visualization_tools/camera_snapshots/

应用场景：
    1. 验证相机连接和配置
    2. 检查图像质量（亮度、曝光、焦距）
    3. 为配置裁剪区域准备素材（配合 test_camera_crop.py）
    4. 调试相机参数（曝光、白平衡等）
    5. 记录当前场景用于后续分析

后续工作流程：
    1. 运行此工具获取快照
    2. 使用 test_camera_crop.py 配置裁剪区域
    3. 将配置复制到 config.py

常见问题排查：
    1. 相机初始化失败：
       - 检查序列号是否正确
       - 使用 rs-enumerate-devices 查看可用相机
       - 确认相机未被其他程序占用
       - 检查 USB 连接稳定性

    2. 图像为空或黑屏：
       - 检查镜头盖是否打开
       - 调整曝光参数
       - 验证光照条件
       - 检查相机焦距

    3. 图像尺寸不符：
       - 验证 dim 配置正确
       - 检查相机是否支持该分辨率
       - 确认没有额外的缩放

    4. 保存失败：
       - 检查目录写入权限
       - 确认磁盘空间充足
       - 验证文件路径有效

注意事项：
    1. 确保相机已正确连接
    2. 首次使用建议逐个相机测试
    3. 注意光照条件对图像质量的影响
    4. 共享相机配置会自动处理，无需手动干预
    5. 快照文件会累积，注意定期清理
"""
import sys
import os
import cv2
import time
from collections import OrderedDict

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))

from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture

# 导入当前配置
from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig

def main():
    config = MarvinUSBEnvConfig()
    caps = OrderedDict()

    print("=" * 70)
    print("相机快照工具 - 初始化相机...")
    print("=" * 70)

    # 初始化相机
    for cam_name, cam_config in config.REALSENSE_CAMERAS.items():
        # 跳过共享相机
        if cam_name == "side_classifier" and "side_policy" in caps:
            print(f"\n[{cam_name}] 共享 side_policy 相机")
            caps[cam_name] = caps["side_policy"]
            continue

        try:
            print(f"\n[{cam_name}]")
            print(f"  序列号: {cam_config['serial_number']}")
            print(f"  分辨率: {cam_config['dim']}")
            print(f"  曝光: {cam_config.get('exposure', 10500)}")

            cap = VideoCapture(
                RSCapture(
                    name=cam_name,
                    serial_number=cam_config['serial_number'],
                    dim=cam_config['dim'],
                    exposure=cam_config.get('exposure', 10500),
                )
            )
            caps[cam_name] = cap
            print(f"  状态: ✓ 成功")
            time.sleep(0.5)  # 每个相机初始化后等待0.5秒


        except Exception as e:
            print(f"  状态: ✗ 失败 - {e}")
            print("\n提示: 请检查:")
            print("  1. 相机是否连接")
            print("  2. 序列号是否正确 (可用 rs-enumerate-devices 查看)")
            print("  3. 其他程序是否占用相机")
            return

    if not caps:
        print("\n✗ 没有可用的相机")
        return

    print("\n" + "=" * 70)
    print(f"✓ 成功初始化 {len(set(caps.values()))} 个相机")
    print("=" * 70)

    # 获取唯一的相机实例
    unique_cams = {}
    for cam_name, cap in caps.items():
        if cap not in unique_cams.values():
            unique_cams[cam_name] = cap

    # 创建保存目录
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n读取并保存图像到 {snapshot_dir}/")
    print("=" * 70)

    try:
        # 等待相机稳定
        print("\n等待相机稳定...")
        time.sleep(1)

        # 读取并保存每个相机的一帧
        for cam_name, cap in unique_cams.items():
            print(f"\n[{cam_name}] 读取帧...")

            try:
                frame = cap.read()  # VideoCapture.read() 只返回 frame，不返回 ret
                if frame is None:
                    print(f"  ✗ 读取失败")
                    continue
            except Exception as e:
                print(f"  ✗ 读取失败: {e}")
                continue

            h, w = frame.shape[:2]
            print(f"  ✓ 读取成功 (尺寸: {w}x{h})")

            # 保存图像
            filename = f"{cam_name}_{timestamp}.jpg"
            filepath = os.path.join(snapshot_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"  ✓ 已保存: {filename}")

        print("\n" + "=" * 70)
        print("✓ 所有相机快照已保存")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")

    finally:
        print("\n关闭相机...")
        for cap in unique_cams.values():
            try:
                cap.close()
            except:
                pass
        print("✓ 完成")


if __name__ == "__main__":
    main()
