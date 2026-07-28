import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")

# 文件标记法：创建标记文件来触发成功
SUCCESS_FLAG_FILE = "/tmp/marvin_success_flag"
RESET_FLAG_FILE = "/tmp/marvin_reset_flag"

def check_success_flag():
    """检查是否存在成功标记文件"""
    if os.path.exists(SUCCESS_FLAG_FILE):
        os.remove(SUCCESS_FLAG_FILE)
        return True
    return False

def check_reset_flag():
    """检查是否存在重置标记文件"""
    if os.path.exists(RESET_FLAG_FILE):
        os.remove(RESET_FLAG_FILE)
        return True
    return False

def main(_):
    # 清理旧标记
    for f in [SUCCESS_FLAG_FILE, RESET_FLAG_FILE]:
        if os.path.exists(f):
            os.remove(f)

    print("=" * 70)
    print("📝 文件标记采集模式")
    print("=" * 70)
    print(f"✅ 标记成功: touch {SUCCESS_FLAG_FILE}")
    print(f"🔄 标记重置: touch {RESET_FLAG_FILE}")
    print("=" * 70)

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False, collect_classifier_data=True)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="Successes")

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

        # 检查成功标记
        if check_success_flag():
            successes.append(transition)
            pbar.update(1)
            print(f"\n✅ [SUCCESS] 成功样本已记录 ({len(successes)}/{success_needed})\n")
        else:
            failures.append(transition)

        # 检查重置标记
        if check_reset_flag() or done or truncated:
            obs, _ = env.reset()
            if check_reset_flag():
                print("\n🔄 [RESET] 手动重置\n")

    pbar.close()

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"\n✅ saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"✅ saved {len(failures)} failure transitions to {file_name}")

if __name__ == "__main__":
    app.run(main)
