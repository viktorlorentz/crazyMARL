from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import jax.numpy as jp

@dataclass
class MultiQuadConfig:
    # Simulation and policy parameters
    policy_freq: float = 250.0
    sim_steps_per_action: int = 1
    episode_length: int = 2048  # Number of steps per episode

    # Noise parameters
    obs_noise: float = 0.0
    act_noise: float = 0.0

    # Quadrotor parameters
    max_thrust_range: float = 0.3
    num_quads: int = 2
    cable_length: float = 0.4
    payload_mass: float = 0.01

    # Target parameters
    target_start_ratio: float = 0.1
    trajectory: Optional[Any] = None

    # Reward coefficients
    reward_coeffs: Dict[str, float] = field(default_factory=lambda: {
        "distance_reward_coef": 0.0,
        "z_distance_reward_coef": 0.0,
        "velocity_reward_coef": 0.0,
        "safe_distance_coef": 1.0,
        "up_reward_coef": 1.0,
        "linvel_reward_coef": 0.0,
        "ang_vel_reward_coef": 0.0,
        "linvel_quad_reward_coef": 1.0,
        "taut_reward_coef": 1.0,
        "collision_penalty_coef": -1.0,
        "out_of_bounds_penalty_coef": -1.0,
        "smooth_action_coef": -1.0,
        "action_energy_coef": 0.0,
    })
