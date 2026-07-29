import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")
flags.DEFINE_boolean("enable_space_key", False, "Enable space key as backup (default: False, pedal-only)")


success_key = False
reset_key = False
shift_pressed = False  # 追踪 Shift 键状态

def on_press(key):
    global success_key, reset_key, shift_pressed
    try:
        print(f"[KEY_DEBUG] 按键捕获: key={key}, str(key)={str(key)}, type={type(key)}")

        # 追踪 Shift 键
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
            shift_pressed = True

        # 脚踏板1: Shift + 左方向键 = 成功
        if key == keyboard.Key.left and shift_pressed:
            success_key = True
            print("🦶 [PEDAL-1] 脚踏板1触发成功标记！(Shift+←)")

        # 脚踏板3: Shift + 右方向键 = 重置
        elif key == keyboard.Key.right and shift_pressed:
            reset_key = True
            print("🦶 [PEDAL-3] 脚踏板3触发重置！(Shift+→)")

        # 空格键支持（受标志位控制）
        elif str(key) == 'Key.space' and FLAGS.enable_space_key:
            success_key = True
            print("🎯 [SPACE] 空格键触发成功标记！")

        elif str(key) == "'r'":  # 按 'r' 键reset
            reset_key = True
            print("🔄 [RESET] R键触发重置！")

    except AttributeError as e:
        print(f"[KEY_ERROR] {e}")

def on_release(key):
    global shift_pressed
    # 释放 Shift 键
    if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
        shift_pressed = False

def main(_):
    global success_key
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("[KEYBOARD] 键盘监听器已启动")
    print("[PEDAL] 🦶 踩脚踏板1 (Shift+←) 标记成功")
    print("[PEDAL] 🦶 踩脚踏板3 (Shift+→) 触发重置")
    if FLAGS.enable_space_key:
        print("[KEYBOARD] ⌨️  空格键备选已启用 (--enable_space_key)")
    else:
        print("[KEYBOARD] ⌨️  空格键已禁用 (使用 --enable_space_key 启用)")
    print("[KEYBOARD] 或按 r 键重置")

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # collect_classifier_data=True: 使用classifier图像但不加载分类器wrapper，避免自动done
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, collect_classifier_data=True)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    while len(successes) < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
            print(f"✅ [SUCCESS] 成功样本已记录 ({len(successes)}/{success_needed})")
        else:
            failures.append(transition)

        if reset_key or done or truncated:
            obs, _ = env.reset()
            if reset_key:
                reset_key = False
                print("🔄 [RESET] 手动重置")

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
