"""
run_multiquad_random_video.py

This script loads the 'multiquad' environment, runs 2500 simulation steps with random actions,
renders the rollout (rendering every few frames), and saves the result as a video file.
"""

import os
import sys
import platform
# add project root so that `import baselines...` works
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if platform.system() == "Linux":
    os.environ["MUJOCO_GL"] = "egl"

# Create a cache directory relative to the current working directory
cache_dir = os.path.join(os.getcwd(), "xla_cache")
os.makedirs(cache_dir, exist_ok=True)

# Set the XLA cache directory to this folder
os.environ["XLA_CACHE_DIR"] = cache_dir

# Nvlink version mismatch fix
# This is a workaround for a known issue with JAX and NVLink on some systems.
# os.environ["TF_USE_NVLINK_FOR_PARALLEL_COMPILATION"] = "0"
# os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax
import jax.numpy as jp
import imageio
import mujoco  # Used to create an OpenGL context
from brax import envs
import jaxmarl
import time
import wandb
import yaml
# Import training utilities and network definitions from ippo_ff_mabrax.py
from baselines.IPPO.ippo_ff_mabrax import make_train, ActorCritic,CriticModule, ActorModule, batchify, unbatchify

import onnx
from jax2onnx import to_onnx, onnx_function

import tensorflow as tf
import numpy as np

# Set JAX cache
jax.config.update("jax_compilation_cache_dir", cache_dir)
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


def render_video(rollout, env, render_every=10, width=1280, height=720):
    # Create an OpenGL context for rendering
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    print("Starting rendering...")
    frames = env.render(rollout[::render_every], camera="track", width=width, height=height)
    fps = float(1.0 / (env.dt * render_every))
    # Changed video filename as per previous code
    video_filename = "trained_policy_video.mp4"
    imageio.mimsave(video_filename, frames, fps=fps)
    # New wandb logging for the video
    wandb.log({"trained_policy_video": wandb.Video(video_filename, format="mp4")})
    print(f"Video saved to {video_filename}")


def eval_results(eval_env, jit_reset, jit_inference_fn, jit_step):
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from PIL import Image
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap

    # --------------------
    # Simulation
    # --------------------
    n_steps = 2500
    render_every = 2
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state["pipeline_state"]]
    quad_actions_list = []
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state["obs"], act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state["pipeline_state"])
        quad_actions_list.append(np.concatenate([np.array(val) for val in ctrl.values()]))
    # Skipping video rendering since it is handled separately.
    
    # Histogram plot over quad actions.
    quad_actions_flat = np.concatenate(quad_actions_list).flatten()
    quad_actions_flat = 0.5 * (quad_actions_flat + 1)
    plt.figure()
    plt.hist(quad_actions_flat, bins=50)
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Quad Actions')
    plt.savefig('quad_actions_histogram.png')
    print("Plot saved: quad_actions_histogram.png")
    wandb.log({"quad_actions_histogram": wandb.Image('quad_actions_histogram.png')})
    plt.close()


def deep_merge(default: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(default.get(k), dict):
            default[k] = deep_merge(default[k], v)
        else:
            default[k] = v
    return default


def main(config_file=None):
    # Load default config and merge any runner override
    default_path = os.path.join(project_root, "crazymarl", "configs", "defaults", "training.yaml")
    with open(default_path, "r") as f:
        default_cfg = yaml.safe_load(f)["training"]
    if config_file:
        with open(config_file, "r") as f:
            override_cfg = yaml.safe_load(f)["training"]
        config = deep_merge(default_cfg, override_cfg)
    else:
        config = default_cfg
    # Append timestamp to experiment name
    config["NAME"] = f"{config.get('NAME', 'quad_marl')}_{int(time.time())}"

    wandb.init(
        name=config["NAME"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    # Deep-merge wandb overrides into the default config (preserve defaults)
    config = deep_merge(config, dict(wandb.config))

    print("Timesteps:", config["TOTAL_TIMESTEPS"])

    # terminate if num_steps*num_envs is too large, because of the GPU memory
    if config["NUM_STEPS"] * config["NUM_ENVS"] > 2048*2048:
        raise ValueError("NUM_STEPS * NUM_ENVS is too large. Please reduce them.")

    
    
    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_train = jax.random.split(rng)
    
    # Train the policy using IPPO training routine from ippo_ff_mabrax.py
    train_fn = jax.jit(make_train(config, rng_train))
    out = train_fn(rng)
    # Extract the trained train_state (the first element of runner_state)
    train_state = out["runner_state"][0]
    
    # Create the environment for simulation
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Initialize the ActorCritic network with proper dimensions (use first agent's spaces)
    obs_shape = env.observation_spaces[env.agents[0]].shape[0]
    act_dim = env.action_spaces[env.agents[0]].shape[0]

    # create a dummy input for initializing modules
    dummy_obs = jp.zeros((1, obs_shape))

    # Initialize actor
    actor = ActorModule(
        action_dim=act_dim,
        activation=config["ACTIVATION"],
        actor_arch=config.get("ACTOR_ARCH", [128, 64, 64])
    )

    actor_params = train_state.params["params"]["actor_module"]
    actor = actor.bind({'params': actor_params})

    # Initialize critic
    critic = CriticModule(
        activation=config["ACTIVATION"],
        critic_arch=config.get("CRITIC_ARCH", [128, 128, 128])
    )
    critic_params = train_state.params["params"]["critic_module"]
    critic = critic.bind({'params': critic_params})

    def policy_fn(obs, key):
        batched_obs = batchify(obs, env.agents, env.num_agents)
        actions = actor(batched_obs)
        unbatched = unbatchify(actions, env.agents, 1, env.num_agents)
        return {a: jp.squeeze(v, axis=0) for a, v in unbatched.items()}

    # Simulation: run an episode using the trained policy
    sim_steps = 2000
    rng, rng_sim = jax.random.split(rng)
    state = env.reset(rng_sim)
    rollout = [state[1]]
    
    print("Starting simulation with trained policy...")
    for i in range(sim_steps):
        rng, key = jax.random.split(rng)
        actions = policy_fn(env.get_obs(state[1]), key)
        rng, key = jax.random.split(rng)
        _, new_state, rewards, dones, info = env.step_env(key, state[1], actions)
        rollout.append(new_state)
        # If any episode terminates, reset the environment and log the new state
        if any(dones.values()):
            rng, reset_key = jax.random.split(rng)
            state = env.reset(reset_key)
            rollout.append(state[1])
        else:
            state = (None, new_state)
    state = jax.block_until_ready(state)
    print("Simulation finished.")
    
    # Call the separated video rendering function
    render_video(rollout, env)
    
    # Build tf
    class TFActor(tf.keras.Model):
        def __init__(self, actor_arch, action_dim):
            super().__init__()
            self.layers_list = []
            for units in actor_arch:
                self.layers_list.append(tf.keras.layers.Dense(units, activation='tanh'))
            self.layers_list.append(tf.keras.layers.Dense(action_dim, activation=None))

        def call(self, x):
            y = x
            for layer in self.layers_list:
                y = layer(y)
            return y

    # Instantiate and build model
    tf_actor = TFActor(config.get("ACTOR_ARCH", [64,64,64]), act_dim)
    dummy = tf.zeros((1, obs_shape), dtype=tf.float32)
    _ = tf_actor(dummy)  # builds weights

    # Assign JAX-trained weights into TF model
    dense_keys = sorted([k for k in actor_params.keys() if k.startswith("Dense_")],
                        key=lambda x: int(x.split("_")[1]))
    for idx, key in enumerate(dense_keys):
        w = np.array(actor_params[key]["kernel"])
        b = np.array(actor_params[key]["bias"])
        tf_actor.layers_list[idx].set_weights([w, b])

    # Convert to TFLite and save
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_actor)
    tflite_model = converter.convert()
    with open("actor_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model to actor_model.tflite")
    wandb.save("actor_model.tflite")

    # ---- Call eval_results ----
    def dummy_jit_reset(rng):
        s = env.reset(rng)
        # Return a dictionary with valid JAX types.
        return {"pipeline_state": s[1], "obs": env.get_obs(s[1])}

    jit_reset = dummy_jit_reset
    jit_inference_fn = lambda obs, key: (policy_fn(obs, key), None)
    def dummy_jit_step(s, ctrl):
        result = env.step_env(jax.random.PRNGKey(0), s["pipeline_state"], ctrl)
        new_state = result[1]
        new_obs = env.get_obs(new_state)
        return {"pipeline_state": new_state, "obs": new_obs}
    jit_step = dummy_jit_step
    eval_results(env, jit_reset, jit_inference_fn, jit_step)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to YAML config file, overrides the default"
    )
    args = parser.parse_args()
    main(config_file=args.config_file)