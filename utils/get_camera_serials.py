#!/usr/bin/env python3
"""
获取 RealSense 相机序列号

用途：
1. 列出所有连接的 RealSense 相机
2. 显示相机序列号、型号、分辨率等信息
3. 生成相机配置代码

使用方法：
    python get_camera_serials.py

作者: Claude
日期: 2024-07-09
"""

import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("❌ pyrealsense2 未安装")
    print("\n安装方法:")
    print("  pip install pyrealsense2")
    sys.exit(1)

def get_camera_info():
    """获取所有相机信息"""
    context = rs.context()
    devices = context.query_devices()

    if len(devices) == 0:
        print("❌ 未检测到 RealSense 相机")
        print("\n请检查:")
        print("  1. 相机是否连接")
        print("  2. USB 线是否正常")
        print("  3. 是否有权限访问 USB 设备")
        print("\n如果使用虚拟机，需要:")
        print("  - 在虚拟机设置中添加 USB 设备")
        print("  - 确保 USB 控制器版本正确（USB 3.0）")
        return []

    cameras = []

    for i, device in enumerate(devices):
        camera_info = {
            'index': i,
            'serial': device.get_info(rs.camera_info.serial_number),
            'name': device.get_info(rs.camera_info.name),
            'firmware': device.get_info(rs.camera_info.firmware_version),
            'usb_type': device.get_info(rs.camera_info.usb_type_descriptor),
        }

        # 获取支持的分辨率
        camera_info['streams'] = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() == rs.stream.color:
                    video_profile = profile.as_video_stream_profile()
                    resolution = f"{video_profile.width()}x{video_profile.height()}"
                    fps = video_profile.fps()
                    camera_info['streams'].append({
                        'type': 'Color',
                        'resolution': resolution,
                        'fps': fps,
                    })

        cameras.append(camera_info)

    return cameras

def display_camera_info(cameras):
    """显示相机信息"""
    print("=" * 60)
    print(f"检测到 {len(cameras)} 个 RealSense 相机")
    print("=" * 60)

    for camera in cameras:
        print(f"\n[相机 {camera['index'] + 1}]")
        print(f"  型号: {camera['name']}")
        print(f"  序列号: {camera['serial']}")
        print(f"  固件版本: {camera['firmware']}")
        print(f"  USB 类型: {camera['usb_type']}")

        # 显示常用分辨率
        unique_resolutions = set()
        for stream in camera['streams']:
            if stream['type'] == 'Color':
                unique_resolutions.add(stream['resolution'])

        if unique_resolutions:
            print(f"  支持分辨率: {', '.join(sorted(unique_resolutions))}")

def generate_config_code(cameras):
    """生成配置代码"""
    if not cameras:
        return

    print("\n" + "=" * 60)
    print("配置代码")
    print("=" * 60)

    print("\n# 复制以下代码到你的 config.py:")
    print("\nREALSENSE_CAMERAS = {")

    for i, camera in enumerate(cameras):
        camera_name = f"wrist_{i+1}" if i < 2 else f"side_{i-1}"

        print(f'    "{camera_name}": {{')
        print(f'        "serial_number": "{camera["serial"]}",')
        print(f'        "dim": (1280, 720),  # 或 (640, 480)')
        print(f'        "exposure": 10500,   # 根据光照条件调整')
        print(f'    }},')

    print("}")

    print("\n# 图像裁剪（需要根据实际视野调整）")
    print("IMAGE_CROP = {")
    for i, camera in enumerate(cameras):
        camera_name = f"wrist_{i+1}" if i < 2 else f"side_{i-1}"
        print(f'    "{camera_name}": lambda img: img[50:-200, 200:-200],  # 需要调整')
    print("}")

def test_camera_stream(serial_number):
    """测试相机流"""
    print(f"\n测试相机 {serial_number} 的视频流...")

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        print("  启动相机...")
        pipeline.start(config)

        print("  获取 10 帧...")
        for i in range(10):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                print(f"    帧 {i+1}/10 - OK")
            else:
                print(f"    帧 {i+1}/10 - 失败")

        pipeline.stop()
        print("  ✅ 相机流测试成功")
        return True

    except Exception as e:
        print(f"  ❌ 相机流测试失败: {e}")
        return False

def main():
    print("=" * 60)
    print("RealSense 相机检测工具")
    print("=" * 60)

    # 获取相机信息
    cameras = get_camera_info()

    if not cameras:
        return

    # 显示信息
    display_camera_info(cameras)

    # 生成配置
    generate_config_code(cameras)

    # 测试相机
    print("\n" + "=" * 60)
    print("相机流测试")
    print("=" * 60)

    test_all = input("\n是否测试所有相机的视频流? (y/n): ").strip().lower()

    if test_all == 'y':
        for camera in cameras:
            success = test_camera_stream(camera['serial'])
            if not success:
                print(f"\n⚠️  相机 {camera['serial']} 测试失败")
                print("  可能原因:")
                print("  1. 相机被其他程序占用")
                print("  2. USB 带宽不足（尝试降低分辨率）")
                print("  3. 驱动问题")

    print("\n" + "=" * 60)
    print("✅ 检测完成")
    print("=" * 60)
    print("\n接下来的步骤:")
    print("1. 复制上面的配置代码到 config.py")
    print("2. 根据实际安装位置命名相机 (wrist_1, side_policy 等)")
    print("3. 调整图像裁剪区域 (IMAGE_CROP)")
    print("4. 运行训练脚本测试相机是否正常工作")

if __name__ == "__main__":
    main()
