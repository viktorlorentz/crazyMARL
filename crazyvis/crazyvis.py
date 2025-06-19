#!/usr/bin/env python3
import os
import argparse
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
def render_video(pipeline_states, env, fps=25, width=1920, height=1080, output="rollout_video.mp4"):    
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    # Prepare directories for images
    video_dir = os.path.dirname(output)
    base_dir = os.path.dirname(video_dir)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print("Rendering rollout...")
    # Save first frame as PNG to images directory
    first_frame = env.render(pipeline_states[:1], camera="track", width=width, height=height)[0]
    png_name = os.path.join(images_dir, os.path.splitext(os.path.basename(output))[0] + "_first_frame.png")
    imageio.imwrite(png_name, first_frame)
    print(f"First frame saved to {png_name}")
    # calculate render_every based on fps and env.dt
    render_every = int(np.round(1.0 / (env.dt * fps)))  

    frames = env.render(pipeline_states[::render_every], camera="track", width=width, height=height)
    fps = float(1.0 / (env.dt * render_every))
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
def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multiquad rollout with a TFLite actor, save to ASDF, and render video"
    )
    p.add_argument("--model_path", type=str, default="actor_model.tflite", help="Path to TFLite actor model")
    p.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments")
    p.add_argument("--timesteps", type=int, default=4000, help="Number of simulation steps")
    p.add_argument("--output", type=str, default="flights.crazy.asdf", help="ASDF output filename")
    p.add_argument("--video", type=str, default="rollout_video.mp4", help="Rendered video filename")
    p.add_argument("--experiment_name", type=str, default="experiment", help="Experiment name prefix")
    return p.parse_args()

#------------------------------------------------------------------------------
def load_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter

#------------------------------------------------------------------------------
def run_batched_rollout(interpreter: tf.lite.Interpreter, env, num_envs: int, env_config: dict):
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]
    timesteps = env_config["episode_length"]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [num_envs, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    # Batch reset to get initial states
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, num_envs)
    obs_batched, pipeline_states = jax.vmap(env.reset)(keys)

    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    vm_step = jax.vmap(env.step_env, in_axes=(0, 0, 0))

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
        obs_batched, pipeline_states, rewards, dones, _ = vm_step(
            step_keys, pipeline_states, raw_acts
        )
        rew_buf.append(np.array(rewards["__all__"]))
        done_buf.append(np.array(dones["__all__"]))

    return (
        np.stack(obs_buf),
        np.stack(act_buf),
        np.stack(rew_buf),
        np.stack(done_buf),
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

    rng = jax.random.PRNGKey(1)
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
        if any(dones.values()):
            print("Resetting environment due to done state")
            rng, key = jax.random.split(rng)
            obs, state = env.reset(key)
    return states

#------------------------------------------------------------------------------
def save_rollout(obs_h, act_h, rew_h, done_h, args, num_envs, agents, env_config):
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
                "model_path": args.model_path,
                "env": "multiquad_ix4",
                "env_config": env_config,          
            },
            "agents": agents_dict,
            "global": {
                "rewards": rew_h,
                "dones": done_h,
                "trajectory": env_config.get("trajectory")  # keep trajectory field
            },
        }]
    }
    asdf.AsdfFile(tree).write_to(args.output)
    print(f"Saved ASDF to {args.output}")


def figure_eight(length, width=1.0, height=1.0, rounds=1, z= np.array([1.5])):
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
    base_dir = os.path.join("experiments", experiment_name + timestamp)
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
    obs_h, act_h, rew_h, done_h, agents = run_batched_rollout(
        interpreter, env, num_envs, env_config
    )
    save_rollout(
        obs_h, act_h, rew_h, done_h,
        argparse.Namespace(model_path=model_path, output=asdf_file),
        num_envs, agents, env_config
    )

    # Single-env rollout + render
    print("Running single-env rollout for rendering...")
    states = run_single_rollout(interpreter, env, env_config)
    render_video(states, env, width=1920, height=1080, output=video_file)

    # print list of all files created
    print("Experiment files created:")
    print(f"  Experiment directory: {base_dir}")
    print(f"  ASDF file: {asdf_file}")
    print(f"  Video file: {video_file}")

    # return the directory for further processing if needed
    return base_dir




def main():
    args = parse_args()

    default_config = {
        "policy_freq": 250.0,              # Policy frequency in Hz.
        "sim_steps_per_action": 1,         # Physics steps between control actions.
        "obs_noise": 0.0,                  # Parameter for observation noise
        "act_noise": 0.0,                  # Parameter for actuator noise
        "max_thrust_range": 0.2,           # Range for randomizing thrust
        "num_quads": 2,
        "cable_length": 0.4,               # Length of the cable connecting the payload to the quadrotors.
        "trajectory": None,                # Array of target positions for the payload
        "target_start_ratio": 0.2,         # Percentage of resets to target position
        "payload_mass": 0.01,              # Mass of the payload.
    }


    
    two_quads_figure_eight_config = {
        **default_config,
        "episode_length": 5000,
        "trajectory": figure_eight(5000),
        "num_quads": 2,
        "target_start_ratio": 1.0,  # Start at the target position
    }

    record_experiment(
        experiment_name="2_quads_figure_eight",
        model_path=args.model_path,
        num_envs=100,
        env_config=two_quads_figure_eight_config,
    )
 
if __name__ == "__main__":
    main()


