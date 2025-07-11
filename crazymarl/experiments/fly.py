#!/usr/bin/env python3
import os
import sys

# ensure crazymarl/ (the package root) is on PYTHONPATH before any jaxmarl imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import jaxmarl
import asdf
import imageio
import mujoco  # for OpenGL context
import datetime
import shutil
import argparse
from omegaconf import OmegaConf

import crazymarl.envs

SEED= 42

#------------------------------------------------------------------------------
# Vendored batchify / unbatchify (no external deps)
def batchify(x: dict, agent_list, num_actors):
    max_dim = max(x[a].shape[-1] for a in agent_list)
    def pad(z):
        return jnp.concatenate(
            [z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))],
            axis=-1
        )
    stacked = jnp.stack([
        x[a] if x[a].shape[-1] == max_dim else pad(x[a])
        for a in agent_list
    ])
    return stacked.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    split = x.reshape((len(agent_list), num_envs, -1))
    return {agent_list[i]: split[i] for i in range(len(agent_list))}

#------------------------------------------------------------------------------
# Rendering utility
def render_video(state, env, env_config, fps=25, width=1920, height=1080, output="rollout_video.mp4"):    
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    # Prepare directories for images
    video_dir = os.path.dirname(output)
    base_dir = os.path.dirname(video_dir)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print("Rendering rollout...")
    # Save first frame as PNG to images directory
    first_frame = env.render(state[:1], camera="track", width=width, height=height)[0]
    png_name = os.path.join(images_dir, os.path.splitext(os.path.basename(output))[0] + "_first_frame.png")
    imageio.imwrite(png_name, first_frame)
    print(f"First frame saved to {png_name}")
  
    render_every = int(np.round(env_config["policy_freq"] / fps))

    frames = env.render(state[::render_every], camera="track", width=width, height=height)
    imageio.mimsave(output, frames, fps=fps)
    print(f"Video saved to {output}")

#------------------------------------------------------------------------------
# Setup XLA/MuJoCo cache
env_cache = os.path.join(os.getcwd(), "xla_cache")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.makedirs(env_cache, exist_ok=True)
os.environ["XLA_CACHE_DIR"] = env_cache
jax.config.update("jax_compilation_cache_dir", env_cache)

#------------------------------------------------------------------------------
def load_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter

#------------------------------------------------------------------------------
def run_batched_rollout(interpreter: tf.lite.Interpreter, env, num_envs: int, env_config: dict):
    ids = env.env.ids # get mjx ids for the environment
    
    print(f"ENV IDS: {env.env.ids}")

    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]
    print(f"Running batched rollout with {num_envs} environments and {len(agents)} agents.")
    print(f"Observation dimension: {obs_dim}")
    timesteps = env_config["episode_length"]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [num_envs, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    # Batch reset to get initial states
    rng = jax.random.PRNGKey(SEED)
    keys = jax.random.split(rng, num_envs)
    obs_batched, state = jax.vmap(env.reset)(keys)

    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    vm_step = jax.vmap(env.step_env, in_axes=(0, 0, 0))

    payload_buf = []  # <-- new: collect payload positions each step
    for _ in range(timesteps):
        # Record observations
        obs_stack = np.stack([np.array(obs_batched[a]) for a in agents], axis=1)
        obs_buf.append(obs_stack)

        # Actor inference
        raw_acts = {}
        for a in agents:
            inp = np.array(obs_batched[a], dtype=np.float32)
            interpreter.set_tensor(inp_idx, inp)
            interpreter.invoke()
            raw_acts[a] = jnp.array(interpreter.get_tensor(out_idx))
        act_stack = np.stack([np.array(raw_acts[a]) for a in agents], axis=1)
        act_buf.append(act_stack)

        # Step environments
        rng, *step_keys = jax.random.split(rng, num_envs + 1)
        step_keys = jnp.stack(step_keys)
        obs_batched, state, rewards, dones, _ = vm_step(
            step_keys, state, raw_acts
        )

        rew_buf.append(np.array(rewards["__all__"]))
        done_buf.append(np.array(dones["__all__"]))
        # record payload_pos for all envs
        payload_buf.append(
            np.array(state.pipeline_state.xpos[:, ids["payload_body_id"], :])
        )

    return (
        np.stack(obs_buf),
        np.stack(act_buf),
        np.stack(rew_buf),
        np.stack(done_buf),
        np.stack(payload_buf),
        agents
    )

#------------------------------------------------------------------------------
def run_single_rollout(interpreter: tf.lite.Interpreter, env, env_config: dict):
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]
    timesteps = env_config["episode_length"]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [1, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    rng = jax.random.PRNGKey(SEED)
    obs, state = env.reset(rng)
    states = []
    for _ in range(timesteps):
        states.append(state)
        # Per-agent inference
        actions = {}
        for a in agents:
            ob = np.expand_dims(obs[a], axis=0).astype(np.float32)
            interpreter.set_tensor(inp_idx, ob)
            interpreter.invoke()
            out = interpreter.get_tensor(out_idx)
            actions[a] = jnp.squeeze(out, axis=0)
        # Step single environment
        rng, key = jax.random.split(rng)
        obs, state, _, dones, _ = env.step_env(key, state, actions)
        # if any done
        # if any(dones.values()):
        #     print("Resetting environment due to done state")
        #     rng, key = jax.random.split(rng)
        #     obs, state = env.reset(key)
    return states

#------------------------------------------------------------------------------
def save_rollout(obs_h, act_h, rew_h, done_h, payload_h, model_path, output, num_envs, agents, env_config):
    agents_dict = {
        agents[i]: {"observations": obs_h[:, :, i, :], "actions": act_h[:, :, i, :]}
        for i in range(len(agents))
    }
    tree = {
        "metadata": {
            "num_envs": num_envs,
            "timesteps": obs_h.shape[0],
            "agent_names": agents,
        },
        "environment": {"name": "multiquad_ix4"},
        "flights": [{
            "metadata": {
                "num_envs": num_envs,
                "timesteps": obs_h.shape[0],
                "agents": agents,
                "model_path": model_path,
                "env": "multiquad_ix4",
                "env_config": env_config,          
            },
            "agents": agents_dict,
            "global": {
                "rewards": rew_h,
                "dones": done_h,
                "trajectory": env_config.get("trajectory"),  # keep trajectory field
                "state": {
                    "payload_pos": payload_h,  # storing payload positions
                }
            },
        }]
    }
    asdf.AsdfFile(tree).write_to(output)
    print(f"Saved ASDF to {output}")


def figure_eight(length, width=1.0, height=1.0, rounds=1, z = np.array([1.5])):
    """
    Generate a figure-eight trajectory.

    Parameters:
    - length:   number of sampling points (timesteps).
    - width:    total width (peak-to-peak) of the figure-8 in x.
    - height:   total height (peak-to-peak) of the figure-8 in y.
    - rounds:   number of complete figure-8 loops over the span.

    Returns:
    - path:        a numpy array of shape (length, 3) representing the trajectory in 3D space.
    """
    # directly sample the parameter from 0 to 2π·rounds
    t = np.linspace(0, 2 * np.pi * rounds, length)
    x = width * np.sin(t)
    y = height * np.sin(2 * t)

    #expand z to match the shape of x and y if not already same length
    if z.shape[0] == 1:
        z = np.full_like(x, z[0])
    elif z.shape[0] != length:
        raise ValueError(f"z must be of length 1 or {length}, got {len(z)}")
    
    return np.column_stack((x, y, z))


   
    
    

#------------------------------------------------------------------------------
def record_experiment(
    experiment_name: str,
    model_path: str,
    num_envs: int,
    env_config: dict,
):
    # prepare experiment directory
    timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    base_dir = os.path.join("experiments_data", experiment_name + timestamp)
    for sub in ["videos", "images", "plots", "policies"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    # copy policy file into policies folder
    policy_dest = os.path.join(base_dir, "policies", os.path.basename(model_path))
    shutil.copy(model_path, policy_dest)
    print(f"Copied policy file to {policy_dest}")

    # derive filenames from experiment_name
    asdf_file = os.path.join(base_dir, f"{experiment_name}.crazy.asdf")
    video_file = os.path.join(base_dir, "videos", f"{experiment_name}.mp4")

    interpreter = load_model(model_path)

    print(f"Creating environment with config: {env_config}")
    env = jaxmarl.make("multiquad_ix4", **env_config)

    # Batched rollout and save
    obs_h, act_h, rew_h, done_h, payload_h, agents = run_batched_rollout(
        interpreter, env, num_envs, env_config
    )
    save_rollout(
        obs_h, act_h, rew_h, done_h, payload_h,
        model_path, asdf_file, num_envs, agents, env_config
    )

    # Single-env rollout + render
    print("Running single-env rollout for rendering...")
    states = run_single_rollout(interpreter, env, env_config)
    render_video(states, env, env_config, width=1920, height=1080, output=video_file)

    # print list of all files created
    print("Experiment files created:")
    print(f"  Experiment directory: {base_dir}")
    print(f"  ASDF file: {asdf_file}")
    print(f"  Video file: {video_file}")

    # return the directory for further processing if needed
    return base_dir


def main():
    parser = argparse.ArgumentParser(description="Run flight experiment from YAML config")
    # accept both --num-quads and --num_quads
    parser.add_argument("--config", type=str, required=True,
                        help="Config file name (without extension) or path to YAML")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to TFLite model file (defaults to trained_policies based on num_quads)")
    parser.add_argument("--num-quads", "--num_quads", dest="num_quads",
                        type=int, required=False,
                        help="Override flights.num_quads")
    args, unknown = parser.parse_known_args()

    config_dir = os.path.join(os.getcwd(), "crazymarl", "experiments", "configs")
    default_path = os.path.join(config_dir, "default.yaml")
    default_conf = OmegaConf.load(default_path)
    cfg = args.config
    config_path = cfg if os.path.isabs(cfg) else os.path.join(config_dir, f"{cfg}.yaml")
    custom_conf = OmegaConf.load(config_path)
    merged_conf = OmegaConf.merge(default_conf, custom_conf)

    # merge CLI overrides
    if unknown:
        cli_conf = OmegaConf.from_cli(unknown)
        merged_conf = OmegaConf.merge(merged_conf, cli_conf)

    # ensure num_quads is set
    if args.num_quads is not None:
        merged_conf.flights.num_quads = args.num_quads
    if not hasattr(merged_conf.flights, 'num_quads'):
        raise ValueError("Please specify --num-quads or add num_quads under 'flights:' in your config.")

    n = merged_conf.flights.num_quads
    # load per-quad overrides
    quad_dir = os.path.join(config_dir, "num_quads")
    quad_path = os.path.join(quad_dir, f"{n}.yaml")
    if os.path.isfile(quad_path):
        override_conf = OmegaConf.load(quad_path)
        merged_conf = OmegaConf.merge(merged_conf, override_conf)

    container = OmegaConf.to_container(merged_conf, resolve=True)
    env_config = container['flights']

    traj_type = env_config.get('trajectory_type')
    if traj_type == 'figure_eight':
        env_config['trajectory'] = figure_eight(env_config['episode_length'])
    elif traj_type == 'recovery':
        env_config['trajectory'] = np.array([0, 0, 1.5])
    env_config.pop('trajectory_type', None)

    num_envs = env_config.get('num_envs')
    env_config.pop('num_envs', None)
    model_path = args.model_path or os.path.join(os.getcwd(), "trained_policies", f"{n}_quad_policy.tflite")
    experiment_name = f"{n}_quads_{cfg}"
    record_experiment(experiment_name, model_path, num_envs, env_config)

if __name__ == "__main__":
    main()