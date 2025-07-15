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

    # os.environ["JAX_PLATFORM_NAME"] = "CPU"  # Force JAX to use CPU
    # jax.config.update("jax_platform_name", "cpu")  # Ensure JAX uses CPU

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


