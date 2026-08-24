#!/usr/bin/env python3
"""验证左右脚踏板是否分别发送 Shift+Left 和 Shift+Right。"""


def main():
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise SystemExit(f"无法加载 pynput: {exc}")
    except Exception as exc:
        raise SystemExit(f"无法监听键盘: {exc}")

    shift_keys = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}
    directions = {keyboard.Key.left: "左踏板", keyboard.Key.right: "右踏板"}
    shifts_down = set()
    pedals_down = set()
    completed = set()

    print("脚踏板测试开始（按 Esc 可退出）")
    print("请分别踩下并松开左、右踏板；预期为 Shift+← / Shift+→。\n")

    def on_press(key):
        if key in shift_keys:
            shifts_down.add(key)
        elif key in directions and key not in pedals_down:
            if not shifts_down:
                print(f"❌ {directions[key]}方向键已收到，但缺少 Shift", flush=True)
                return
            pedals_down.add(key)
            print(f"✅ {directions[key]}：按下", flush=True)

    def on_release(key):
        if key == keyboard.Key.esc:
            print("\n测试已取消。")
            return False
        if key in directions and key in pedals_down:
            pedals_down.remove(key)
            completed.add(key)
            print(f"✅ {directions[key]}：松开", flush=True)
            if completed == set(directions):
                print("\n🎉 测试通过：左右踏板均可正确识别。")
                return False
        if key in shift_keys:
            shifts_down.discard(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
