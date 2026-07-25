#!/usr/bin/env python3
"""
相机裁剪区域配置工具 - Camera Crop Configuration Tool

功能：
    交互式工具用于配置 RealSense 相机的图像裁剪区域。
    通过可视化界面选择裁剪区域，自动生成配置代码。

主要功能：
    1. 从保存的相机快照加载图像
    2. 交互式选择裁剪区域（鼠标拖动）
    3. 实时预览裁剪效果
    4. 自动生成 Python 配置代码
    5. 支持多相机配置

工作流程：
    1. 准备工作：
       - 先运行 view_cameras.py 生成相机快照
       - 快照保存在 utils/visualization_tools/camera_snapshots/

    2. 启动工具：
       - 自动查找最新的相机快照
       - 为每个相机创建独立窗口
       - 加载图像并显示

    3. 选择裁剪区域（对每个相机）：
       a) 鼠标操作：
          - 按住左键拖动：绘制裁剪矩形
          - 矩形会实时显示在图像上（绿色边框）
       b) 实时预览：
          - 按 'p'：在新窗口预览裁剪后的图像
          - 可多次调整直到满意
       c) 重置：
          - 按 'r'：清除当前相机的裁剪区域

    4. 保存配置：
       - 按 's'：保存所有相机的裁剪配置
       - 生成文件：utils/test_tools/camera_crop_config.txt
       - 文件包含可直接复制的 Python 代码

    5. 退出：
       - 按 'q'：退出程序

输出格式（camera_crop_config.txt）：
    # 相机裁剪配置
    # 生成时间: <timestamp>

    CAMERA_CROPS = {
        "camera_name_1": lambda img: img[y1:y2, x1:x2],
        "camera_name_2": lambda img: img[y1:y2, x1:x2],
        ...
    }

配置说明：
    - 裁剪坐标使用 NumPy 切片格式：img[y1:y2, x1:x2]
    - y1, y2: 行方向（高度）的起止位置
    - x1, x2: 列方向（宽度）的起止位置
    - lambda 函数可直接用于 config.py 中

鼠标操作：
    - 左键拖动：选择裁剪区域
    - 绿色矩形：当前选择的区域
    - 红色十字：鼠标位置坐标

键盘操作：
    - 's': 保存配置到文件
    - 'p': 预览裁剪后的图像
    - 'r': 重置当前相机的裁剪
    - 'q': 退出程序

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 1. 先获取相机快照
    python utils/visualization_tools/view_cameras.py

    # 2. 配置裁剪区域
    python utils/test_tools/test_camera_crop.py

    # 3. 查看生成的配置
    cat utils/test_tools/camera_crop_config.txt

    # 4. 将配置复制到 config.py 中使用

典型应用场景：
    1. 去除图像边缘无用区域，减小网络输入尺寸
    2. 聚焦任务关键区域（如抓取目标位置）
    3. 去除固定的遮挡物或标记
    4. 统一多相机的感兴趣区域

注意事项：
    1. 必须先运行 view_cameras.py 生成快照
    2. 裁剪区域应保持任务关键信息
    3. 裁剪后尺寸会影响网络输入，需要调整相应配置
    4. 训练和测试时必须使用相同的裁剪配置
    5. 配置文件会覆盖，建议备份重要配置
"""
import sys
import os
import cv2
import numpy as np
import glob

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from examples.experiments.marvin_usb_insertion.config import MarvinUSBEnvConfig

class CropSelector:
    def __init__(self, window_name, original_image):
        self.window_name = window_name
        self.original = original_image.copy()
        self.display = original_image.copy()
        self.h, self.w = original_image.shape[:2]

        # 裁剪坐标 (x1, y1, x2, y2)
        self.crop_coords = None
        self.drawing = False
        self.start_point = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 实时显示拖动中的矩形和参数
                self.display = self.original.copy()

                # 绘制拖动中的矩形
                cv2.rectangle(self.display, self.start_point, (x, y), (0, 255, 0), 2)

                # 计算临时坐标
                x1, y1 = self.start_point
                x2, y2 = x, y
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                # 限制范围
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.w, x2), min(self.h, y2)

                # 临时设置坐标以显示参数
                temp_coords = self.crop_coords
                self.crop_coords = (x1, y1, x2, y2)
                self.update_display()
                self.crop_coords = temp_coords

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.start_point
            x2, y2 = x, y

            # 确保 x1 < x2, y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # 限制在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.w, x2), min(self.h, y2)

            if x2 - x1 > 10 and y2 - y1 > 10:  # 最小尺寸
                self.crop_coords = (x1, y1, x2, y2)
                self.display = self.original.copy()
                cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.update_display()
                self.print_config()

    def update_display(self):
        # 添加信息文本
        info_img = self.display.copy()

        # 添加半透明背景 - 扩大以容纳更多信息
        overlay = info_img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, info_img, 0.3, 0, info_img)

        # 相机名称
        cv2.putText(info_img, self.window_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 原始图像尺寸
        cv2.putText(info_img, f"Original: {self.w}x{self.h}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.crop_coords:
            x1, y1, x2, y2 = self.crop_coords
            crop_w, crop_h = x2 - x1, y2 - y1
            top, bottom, left, right = y1, self.h - y2, x1, self.w - x2

            # 裁剪后尺寸
            cv2.putText(info_img, f"Cropped: {crop_w}x{crop_h}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 裁剪参数
            cv2.putText(info_img, f"Params: top={top}, bottom={bottom}, left={left}, right={right}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # lambda 表达式
            if bottom == 0 and right == 0:
                config_str = f'img[{top}:, {left}:]'
            elif bottom == 0:
                config_str = f'img[{top}:, {left}:-{right}]'
            elif right == 0:
                config_str = f'img[{top}:-{bottom}, {left}:]'
            else:
                config_str = f'img[{top}:-{bottom}, {left}:-{right}]'

            cv2.putText(info_img, f"Code: {config_str}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        else:
            cv2.putText(info_img, "Drag to select crop area", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(self.window_name, info_img)

    def print_config(self):
        if self.crop_coords:
            x1, y1, x2, y2 = self.crop_coords
            top = y1
            bottom = self.h - y2
            left = x1
            right = self.w - x2

            print(f"\n{'='*70}")
            print(f"[{self.window_name}] 裁剪区域已选择:")
            print(f"{'='*70}")
            print(f"  原始尺寸: {self.w}x{self.h}")
            print(f"  裁剪坐标: ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"  裁剪后尺寸: {x2-x1}x{y2-y1}")
            print(f"  裁剪参数: top={top}, bottom={bottom}, left={left}, right={right}")

            # 生成配置代码
            if bottom == 0 and right == 0:
                config_str = f'lambda img: img[{top}:, {left}:]'
            elif bottom == 0:
                config_str = f'lambda img: img[{top}:, {left}:-{right}]'
            elif right == 0:
                config_str = f'lambda img: img[{top}:-{bottom}, {left}:]'
            else:
                config_str = f'lambda img: img[{top}:-{bottom}, {left}:-{right}]'

            print(f"  配置代码: {config_str}")
            print(f"{'='*70}\n")

    def get_config_dict(self):
        """返回配置字典"""
        if self.crop_coords:
            x1, y1, x2, y2 = self.crop_coords
            top = y1
            bottom = self.h - y2
            left = x1
            right = self.w - x2

            return {
                'top': top,
                'bottom': bottom,
                'left': left,
                'right': right,
                'crop_size': (x2-x1, y2-y1)
            }
        return None

    def reset(self):
        """重置裁剪区域"""
        self.crop_coords = None
        self.display = self.original.copy()
        self.update_display()
        print(f"\n[{self.window_name}] 裁剪区域已重置")

    def show_cropped(self):
        """显示裁剪后的图像"""
        if self.crop_coords:
            x1, y1, x2, y2 = self.crop_coords
            cropped = self.original[y1:y2, x1:x2]

            preview_win = f"{self.window_name}_cropped"
            cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
            cv2.imshow(preview_win, cropped)

def load_camera_images():
    """加载 test_single_camera.py 保存的图像"""
    config = MarvinUSBEnvConfig()
    images = {}

    print("=" * 70)
    print("加载相机快照...")
    print("=" * 70 + "\n")

    # 查找图像文件
    for cam_name in ['wrist_1', 'wrist_2', 'side_policy']:
        # 查找最新的测试图像
        pattern = f"{cam_name}_test.jpg"
        if os.path.exists(pattern):
            img = cv2.imread(pattern)
            if img is not None:
                images[cam_name] = img
                h, w = img.shape[:2]
                print(f"✓ 加载 {cam_name}: {w}x{h} from {pattern}")
            else:
                print(f"✗ 无法读取 {pattern}")
        else:
            print(f"✗ 未找到 {pattern}")
            print(f"  请先运行: python test_single_camera.py")

    # side_classifier 使用 side_policy 的图像
    if 'side_policy' in images:
        images['side_classifier'] = images['side_policy']
        print(f"✓ side_classifier 使用 side_policy 图像")

    return images

def save_config(selectors, config):
    """保存配置到文件"""
    # 保存到 utils/test_tools 目录下
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "camera_crop_config.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 相机裁剪区域配置\n")
        f.write("# 生成时间: " + __import__('time').strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("# 将以下配置复制到 config.py 的 IMAGE_CROP 字典中:\n\n")
        f.write("IMAGE_CROP = {\n")

        for cam_name, selector in selectors.items():
            cfg = selector.get_config_dict()
            if cfg:
                top, bottom, left, right = cfg['top'], cfg['bottom'], cfg['left'], cfg['right']
                crop_w, crop_h = cfg['crop_size']

                f.write(f"    # {cam_name}: 裁剪后尺寸 {crop_w}x{crop_h}\n")

                if bottom == 0 and right == 0:
                    config_str = f'lambda img: img[{top}:, {left}:]'
                elif bottom == 0:
                    config_str = f'lambda img: img[{top}:, {left}:-{right}]'
                elif right == 0:
                    config_str = f'lambda img: img[{top}:-{bottom}, {left}:]'
                else:
                    config_str = f'lambda img: img[{top}:-{bottom}, {left}:-{right}]'

                f.write(f'    "{cam_name}": {config_str},\n')
            else:
                f.write(f'    "{cam_name}": lambda img: img,  # 未设置裁剪\n')

        f.write("}\n")

    print(f"\n✓ 配置已保存到: {output_file}")
    print(f"  请复制其中的配置到 config.py")

def main():
    print("\n" + "=" * 70)
    print("相机裁剪区域配置工具 - 从图片加载")
    print("=" * 70 + "\n")

    print("操作说明:")
    print("  - 鼠标拖动: 选择裁剪区域")
    print("  - 按 's': 保存所有配置到 camera_crop_config.txt")
    print("  - 按 'r': 重置当前相机的裁剪区域")
    print("  - 按 'p': 预览裁剪后的图像")
    print("  - 按 'q': 退出")
    print("=" * 70 + "\n")

    # 加载图像
    images = load_camera_images()

    if not images:
        print("\n✗ 没有找到任何相机图像")
        print("  请先运行: python test_single_camera.py")
        return

    print(f"\n✓ 成功加载 {len(images)} 个相机图像")
    print("=" * 70 + "\n")

    # 创建选择器
    selectors = {}
    for cam_name, image in images.items():
        if cam_name == 'side_classifier':
            continue  # 跳过，因为与 side_policy 共享图像

        selector = CropSelector(cam_name, image)
        selectors[cam_name] = selector
        selector.update_display()

    # side_classifier 单独处理
    if 'side_policy' in images:
        selector = CropSelector('side_classifier', images['side_policy'])
        selectors['side_classifier'] = selector
        selector.update_display()

    print("\n开始配置裁剪区域...")
    print("请在每个窗口中拖动鼠标选择裁剪区域\n")

    config = MarvinUSBEnvConfig()

    try:
        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                print("\n退出...")
                break

            elif key == ord('s'):
                print("\n保存配置...")
                save_config(selectors, config)

            elif key == ord('r'):
                # 重置所有
                for selector in selectors.values():
                    selector.reset()

            elif key == ord('p'):
                # 预览裁剪
                for selector in selectors.values():
                    selector.show_cropped()

    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")

    finally:
        cv2.destroyAllWindows()
        print("✓ 完成")

if __name__ == "__main__":
    main()
