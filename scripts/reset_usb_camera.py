#!/usr/bin/env python3
"""
USB相机软件重置工具

通过unbind/bind USB设备来模拟拔插，解决相机需要重新插拔才能检测的问题
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def find_realsense_devices():
    """查找所有RealSense相机的USB设备路径"""
    devices = []

    # RealSense相机的USB VID:PID
    # Intel RealSense D400系列: 8086:0b07, 8086:0b3a, 8086:0b64等
    realsense_ids = [
    "8086:0ad3",  # D415（你的型号）
    "8086:0b5b",  # D405（你的型号）
    "8086:0b07",  # D435/D435i
    "8086:0b3a",  # D415
    "8086:0b64",  # D405
    "8086:0ad1",  # D455
    "8086:0b5c",  # D435f
]
    try:
        # 查找USB设备
        for usb_device in Path("/sys/bus/usb/devices").iterdir():
            if not usb_device.is_dir():
                continue

            idVendor_file = usb_device / "idVendor"
            idProduct_file = usb_device / "idProduct"

            if not (idVendor_file.exists() and idProduct_file.exists()):
                continue

            vid = idVendor_file.read_text().strip()
            pid = idProduct_file.read_text().strip()
            device_id = f"{vid}:{pid}"

            if device_id in realsense_ids:
                # 获取设备名称
                product_file = usb_device / "product"
                product_name = product_file.read_text().strip() if product_file.exists() else "Unknown"

                devices.append({
                    'path': usb_device,
                    'name': usb_device.name,
                    'vid_pid': device_id,
                    'product': product_name
                })

    except Exception as e:
        print(f"❌ 查找设备时出错: {e}")
        return []

    return devices


def unbind_device(device_path):
    """解绑USB设备"""
    unbind_path = device_path / "driver" / "unbind"

    if not unbind_path.exists():
        print(f"⚠️  设备 {device_path.name} 未绑定驱动")
        return False

    try:
        with open(unbind_path, 'w') as f:
            f.write(device_path.name)
        print(f"✓ 解绑设备: {device_path.name}")
        return True
    except PermissionError:
        print(f"❌ 权限不足，请使用 sudo 运行此脚本")
        return False
    except Exception as e:
        print(f"❌ 解绑失败: {e}")
        return False


def bind_device(device_path):
    """重新绑定USB设备"""
    # 找到对应的驱动
    driver_path = Path(f"/sys/bus/usb/drivers/usb")
    bind_path = driver_path / "bind"

    if not bind_path.exists():
        print(f"❌ 找不到USB驱动绑定接口")
        return False

    try:
        with open(bind_path, 'w') as f:
            f.write(device_path.name)
        print(f"✓ 重新绑定设备: {device_path.name}")
        return True
    except Exception as e:
        print(f"❌ 绑定失败: {e}")
        return False


def reset_usb_device(device_path, delay=2.0):
    """重置USB设备（模拟拔插）"""
    print(f"\n🔄 重置设备: {device_path.name}")

    # 1. 解绑设备
    if not unbind_device(device_path):
        return False

    # 2. 等待
    print(f"⏳ 等待 {delay} 秒...")
    time.sleep(delay)

    # 3. 重新绑定
    if not bind_device(device_path):
        return False

    print(f"✅ 设备重置完成\n")
    return True


def main():
    print("=" * 60)
    print("USB相机软件重置工具")
    print("=" * 60)

    # 检查root权限
    if os.geteuid() != 0:
        print("❌ 需要root权限，请使用 sudo 运行:")
        print(f"   sudo {sys.argv[0]}")
        sys.exit(1)

    # 查找RealSense设备
    print("\n🔍 查找RealSense相机...")
    devices = find_realsense_devices()

    if not devices:
        print("❌ 未找到RealSense相机")
        print("\n提示: 请检查:")
        print("  1. 相机是否已连接")
        print("  2. 使用 'lsusb' 命令查看USB设备")
        sys.exit(1)

    print(f"\n✓ 找到 {len(devices)} 个RealSense设备:\n")

    for i, dev in enumerate(devices, 1):
        print(f"  [{i}] {dev['product']}")
        print(f"      路径: {dev['name']}")
        print(f"      VID:PID: {dev['vid_pid']}")
        print()

    # 重置所有设备
    print("=" * 60)
    success_count = 0

    for dev in devices:
        if reset_usb_device(dev['path'], delay=2.0):
            success_count += 1
        time.sleep(0.5)

    # 总结
    print("=" * 60)
    if success_count == len(devices):
        print(f"✅ 成功重置 {success_count}/{len(devices)} 个设备")
        print("\n💡 相机应该已经可以使用了，请运行你的程序测试")
    else:
        print(f"⚠️  部分重置失败: {success_count}/{len(devices)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
