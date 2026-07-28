#!/usr/bin/env python3
"""测试 pynput 键盘监听"""
from pynput import keyboard
import time

def on_press(key):
    print(f"按键: {key}, str={str(key)}, type={type(key)}")
    try:
        if key == keyboard.Key.space:
            print("✅ 空格键（枚举比较）")
        if str(key) == 'Key.space':
            print("✅ 空格键（字符串比较）")
    except:
        pass

print("开始监听，按任意键测试...")
listener = keyboard.Listener(on_press=on_press)
listener.start()

# 保持运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("退出")
