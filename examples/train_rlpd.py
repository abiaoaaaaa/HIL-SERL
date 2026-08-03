#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string(
    "restore_checkpoint_path", None, "Optional checkpoint directory used only to warm-start training."
)
flags.DEFINE_integer(
    "restore_checkpoint_step", 0, "Checkpoint step to warm-start from; 0 uses the latest checkpoint."
)
flags.DEFINE_boolean(
    "restore_weights_only",
    False,
    "Warm-start network parameters but keep freshly initialized optimizer state.",
)
flags.DEFINE_string(
    "legacy_replay_path", None, "Optional directory containing legacy online transition pickle files."
)
flags.DEFINE_boolean(
    "migrate_legacy_data", False, "Migrate legacy last_action observations and smoothness rewards in memory."
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def clip_position_actions(action: np.ndarray, limit: float) -> np.ndarray:
    """Limit only normalized Cartesian translation actions (dx, dy, dz)."""
    clipped = np.asarray(action, dtype=np.float32).copy()
    clipped[:3] = np.clip(clipped[:3], -limit, limit)
    return clipped


class ContinuousActionEMA:
    """Light EMA filter for continuous arm actions; discrete gripper is untouched."""

    def __init__(self, alpha: float, continuous_dims: int = 4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.continuous_dims = continuous_dims
        self.previous = np.zeros(continuous_dims, dtype=np.float32)

    def reset(self):
        self.previous.fill(0.0)

    def __call__(self, action: np.ndarray) -> np.ndarray:
        filtered = np.asarray(action, dtype=np.float32).copy()
        current = filtered[: self.continuous_dims]
        current = self.alpha * current + (1.0 - self.alpha) * self.previous
        filtered[: self.continuous_dims] = current
        self.previous = current.copy()
        return filtered

    def sync(self, executed_action: np.ndarray):
        """Synchronize after an intervention bypasses the policy filter."""
        self.previous = np.asarray(executed_action, dtype=np.float32)[
            : self.continuous_dims
        ].copy()


def migrate_legacy_transitions(
    transitions,
    previous_executed_action=None,
    ema_alpha=0.65,
):
    """Convert legacy transitions to the current EMA-augmented MDP in memory.

    Replay ``actions`` remain the policy actions emitted before EMA.  The
    ``last_action`` observation must contain the command actually passed to the
    wrapped environment, so legacy sequences are replayed through the same EMA
    recurrence used by the actor.
    """
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError(f"EMA alpha must be in (0, 1], got {ema_alpha}")

    migrated = 0
    for transition in transitions:
        policy_action = np.asarray(transition["actions"], dtype=np.float32)
        obs_state = transition["observations"]["state"]
        next_obs_state = transition["next_observations"]["state"]
        if obs_state.shape[-1] < 6 or next_obs_state.shape[-1] < 6:
            raise ValueError(
                "Legacy transition state must contain last_action at indices [1:6]."
            )

        episode_start = previous_executed_action is None
        if episode_start:
            previous_executed_action = np.zeros_like(policy_action)

        executed_action = policy_action.copy()
        executed_action[:4] = (
            ema_alpha * policy_action[:4]
            + (1.0 - ema_alpha) * previous_executed_action[:4]
        )
        # The gripper bypasses ContinuousActionEMA.
        executed_action[4] = policy_action[4]

        obs_state[..., 1:6] = previous_executed_action
        next_obs_state[..., 1:6] = executed_action

        success = float(transition["rewards"] >= 0.5)
        action_diff = executed_action[:4] - previous_executed_action[:4]
        smoothness_penalty = (
            0.0
            if episode_start
            else 0.01 * float(np.mean(action_diff ** 2))
        )
        transition["rewards"] = success - smoothness_penalty

        previous_executed_action = (
            None if transition["dones"] else executed_action.copy()
        )
        migrated += 1

    return previous_executed_action, migrated


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng, training_start_step=0):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            action_filter = ContinuousActionEMA(config.action_filter_alpha)
            obs, _ = env.reset()
            action_filter.reset()
            episode_finished = False
            start_time = time.time()
            while not episode_finished:
                sampling_rng, key = jax.random.split(sampling_rng)
                policy_actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                    seed=key
                )
                policy_actions = np.asarray(jax.device_get(policy_actions))
                policy_actions = clip_position_actions(
                    policy_actions, config.POSITION_ACTION_LIMIT
                )
                executed_actions = action_filter(policy_actions)

                next_obs, reward, done, truncated, info = env.step(executed_actions)
                if "intervene_action" in info:
                    action_filter.sync(
                        clip_position_actions(
                            info["intervene_action"], config.POSITION_ACTION_LIMIT
                        )
                    )
                obs = next_obs
                episode_finished = done or truncated

                if episode_finished:
                    succeeded = bool(info.get("succeed", reward >= 0.5))
                    if succeeded:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += int(succeeded)
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(
            f"average successful episode time: "
            f"{np.mean(time_list) if time_list else float('nan')}"
        )
        return  # after done eval, return and exit
    
    buffer_files = natsorted(
        glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
    )
    buffer_start_step = (
        int(os.path.basename(buffer_files[-1])[12:-4]) + 1
        if buffer_files
        else 0
    )
    # Checkpoints and buffer chunks may be saved at different cadences (or the
    # learner may continue updating while the actor is offline).  Never move
    # the actor's global step backwards behind the restored network.
    start_step = max(training_start_step, buffer_start_step)

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    action_filter = ContinuousActionEMA(config.action_filter_alpha)
    obs, _ = env.reset()
    action_filter.reset()
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                policy_actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                policy_actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                policy_actions = np.asarray(jax.device_get(policy_actions))

            # Keep replay and execution consistent: constrain only dx/dy/dz
            # before the EMA filter. drz and the discrete gripper are unchanged.
            policy_actions = clip_position_actions(
                policy_actions, config.POSITION_ACTION_LIMIT
            )

            # EMA belongs to the environment/control response. Replay keeps the
            # constrained policy action so SAC trains Q(s, a) on the same action
            # that the actor emits. The executed action is in next_observations.last_action.
            executed_actions = action_filter(policy_actions)

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(executed_actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                buffer_actions = clip_position_actions(
                    info.pop("intervene_action"), config.POSITION_ACTION_LIMIT
                )
                action_filter.sync(buffer_actions)
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                buffer_actions = policy_actions
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=buffer_actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()
                action_filter.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(
    rng,
    agent,
    replay_buffer,
    demo_buffer,
    wandb_logger=None,
    training_start_step=0,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = training_start_step
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Let the actor use the warm-started policy while the learner waits for data.
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            warmup_steps = getattr(config, "critic_warmup_steps", 0)
            in_critic_warmup = step < start_step + warmup_steps
            update_networks = (
                train_critic_networks_to_update
                if in_critic_warmup
                else train_networks_to_update
            )
            agent, update_info = agent.update(
                batch,
                networks_to_update=update_networks,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    training_start_step = 0
    current_checkpoint = (
        checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path)
        else None
    )
    if current_checkpoint is not None:
        if FLAGS.eval_checkpoint_step:
            input(
                "Real-robot evaluation will start after loading "
                f"checkpoint_{FLAGS.eval_checkpoint_step}. Press Enter to continue."
            )
        else:
            input("Checkpoint path already exists. Press Enter to resume training.")
        current_step = FLAGS.eval_checkpoint_step or None
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=current_step,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = (
            str(current_step)
            if current_step is not None
            else os.path.basename(current_checkpoint)[11:]
        )
        training_start_step = int(ckpt_number) + 1
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
    elif FLAGS.restore_checkpoint_path is not None:
        restore_step = FLAGS.restore_checkpoint_step or None
        restore_dir = os.path.abspath(FLAGS.restore_checkpoint_path)
        restored_path = checkpoints.latest_checkpoint(restore_dir)
        if restored_path is None:
            raise FileNotFoundError(f"No checkpoint found in {restore_dir}")
        restored_state = checkpoints.restore_checkpoint(
            restore_dir,
            agent.state,
            step=restore_step,
        )
        if FLAGS.restore_weights_only:
            agent = agent.replace(
                state=agent.state.replace(params=restored_state.params)
            )
        else:
            agent = agent.replace(state=restored_state)
        if restore_step is None:
            restore_step = int(os.path.basename(restored_path)[11:])
        training_start_step = restore_step + 1
        print_green(
            f"Warm-started {'weights only' if FLAGS.restore_weights_only else 'full state'} "
            f"from {restore_dir} at step {restore_step}."
        )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                if FLAGS.migrate_legacy_data:
                    _, count = migrate_legacy_transitions(
                        transitions,
                        ema_alpha=config.action_filter_alpha,
                    )
                    print_green(f"Migrated {count} demo transitions from {path}")
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.legacy_replay_path is not None:
            all_legacy_files = natsorted(
                glob.glob(os.path.join(FLAGS.legacy_replay_path, "transitions_*.pkl"))
            )
            # The replay buffer retains only its newest ``capacity`` samples.
            # Load one extra chunk to preserve action continuity at the first
            # retained chunk without reading tens of GB that would be overwritten.
            chunk_size = max(1, config.buffer_period)
            chunks_to_load = (
                config.replay_buffer_capacity + chunk_size - 1
            ) // chunk_size + 1
            legacy_files = all_legacy_files[-chunks_to_load:]
            previous_action = None
            migrated_count = 0
            for file in legacy_files:
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                if FLAGS.migrate_legacy_data:
                    previous_action, count = migrate_legacy_transitions(
                        transitions,
                        previous_action,
                        ema_alpha=config.action_filter_alpha,
                    )
                    migrated_count += count
                for transition in transitions:
                    replay_buffer.insert(transition)
            print_green(
                f"Loaded newest {len(legacy_files)}/{len(all_legacy_files)} legacy replay files; "
                f"migrated {migrated_count} transitions; buffer size={len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            training_start_step=training_start_step,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
            training_start_step=training_start_step,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
