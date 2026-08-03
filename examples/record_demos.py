import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import json
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
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
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

    # Keep training-condition metadata next to the pickle.  Historical demo
    # files contain only transitions, which makes action/control semantics
    # impossible to verify from the file itself.
    robot_config = env.unwrapped.config
    first_obs = transitions[0]["observations"]
    metadata = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "experiment": FLAGS.exp_name,
        "successful_episodes": success_count,
        "transition_count": len(transitions),
        "action_shape": list(np.asarray(transitions[0]["actions"]).shape),
        "state_shape": list(np.asarray(first_obs["state"]).shape),
        "image_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in first_obs.items()
            if key != "state"
        },
        "control": {
            "robot_ip": robot_config.ROBOT_IP,
            "arm": robot_config.ARM,
            "impedance_mode": robot_config.IMPEDANCE_MODE,
            "action_scale": np.asarray(robot_config.ACTION_SCALE).tolist(),
            "position_action_limit": float(robot_config.POSITION_ACTION_LIMIT),
            "policy_hz": int(robot_config.HZ),
            "stream_control_hz": int(robot_config.STREAM_CONTROL_HZ),
            "stream_linear_velocity_mm_s": float(robot_config.STREAM_LINEAR_VEL_MM_S),
            "stream_linear_acceleration_mm_s2": float(robot_config.STREAM_LINEAR_ACC_MM_S2),
            "stream_yaw_velocity_deg_s": float(robot_config.STREAM_YAW_VEL_DEG_S),
            "stream_yaw_acceleration_deg_s2": float(robot_config.STREAM_YAW_ACC_DEG_S2),
            "compliance_k": np.asarray(robot_config.COMPLIANCE_PARAM["K"]).tolist(),
            "compliance_d": np.asarray(robot_config.COMPLIANCE_PARAM["D"]).tolist(),
            "max_episode_length": int(robot_config.MAX_EPISODE_LENGTH),
        },
    }
    metadata_path = file_name + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"saved demo metadata to {metadata_path}")

if __name__ == "__main__":
    app.run(main)
