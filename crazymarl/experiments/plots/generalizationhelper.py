#!/usr/bin/env python3
import os
import sys

# ensure crazymarl/ (the package root) is on PYTHONPATH before any jaxmarl imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

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

from crazymarl.experiments.fly import load_model, run_batched_rollout

import crazymarl.envs

def run_experiments(
    num_envs: int,
    env_config: dict = None,
):
    if env_config is None:
        env_config = default_env_config

    os.environ["JAX_PLATFORM_NAME"] = "CPU"  # Force JAX to use CPU
    jax.config.update("jax_platform_name", "cpu")  # Ensure JAX uses CPU

    os.chdir(project_root)

    seed = env_config.get("seed", 0)

    env_config.pop("seed", None)
   
    env = jaxmarl.make("multiquad_ix4", **env_config)
    interpreter = load_model("trained_policies/2_quad_policy.tflite")

    # Batched rollout and save
    obs_h, act_h, rew_h, done_h, payload_h, agents = run_batched_rollout(
        interpreter, env, num_envs, env_config, seed=seed
    )
    first_dones = np.argmax(done_h, axis=0)
    full_runs = np.where(first_dones > first_dones.shape[0] - 1)[0]
    # print(f"Number of full runs: {len(full_runs)}")
    # print(f"First dones: {first_dones}")

    return len(full_runs)



default_env_config = {
  
    "num_quads": 2,              # Number of quads in the environment
    "episode_length": 2500,
    "target_start_ratio": 0.0,
    "obs_noise": 0.0,
    "policy_freq": 250.0,         # Policy frequency in Hz
    "sim_steps_per_action": 1,    # Physics steps between control actions
    "obs_noise": 0.0,             # Observation noise parameter
    "act_noise": 0.0,             # Actuator noise parameter
    "max_thrust_range": 0.15,     # Range for randomizing thrust
    "cable_length": 0.3,          # Cable length connecting payload to quads
    "trajectory": None,           # Default trajectory
    "payload_mass": 0.01,         # Mass of the payload
    "auto_reset": False,          # Auto-reset env on done
    "seed": 0,                   
    
}



def eval_generalization(parameter, values, default_env_config=default_env_config):
    config = default_env_config.copy()
    default_val = config[parameter]  # Store the default value

    for v in values:
        config[parameter] = v
        print(f"Running with {parameter}={v}")
        
        # Run the experiments with the modified config
        successful = run_experiments(
            num_envs=1000,
            env_config=config
        )

        rate = successful / 1000.0*100
        print(f"Success rate for {parameter}={v}: {rate:.2f}")
        # save as csv new line
        with open(f"results_generalization.csv", "a") as f:
            f.write(f"{parameter},{v},{rate:.2f},{default_val}\n")


# add command line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generalization in CrazyMarl environments.")
    parser.add_argument("--parameter", type=str, required=True, help="Parameter to vary (e.g., 'max_thrust_range').")
    parser.add_argument("--values", type=float, nargs='+', required=True, help="Values to test for the parameter.")
    
    args = parser.parse_args()
    
    eval_generalization(args.parameter, args.values)


