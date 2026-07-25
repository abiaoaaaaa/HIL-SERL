#!/bin/bash
# RealSense相机快速重置脚本
# 使用方法: ./reset_camera.sh

cd "$(dirname "$0")"
sudo python3 reset_usb_camera.py
