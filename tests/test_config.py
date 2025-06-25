import jax.numpy as jp
from crazymarl.configs.multi_quad_config import MultiQuadConfig

def test_default_config_values():
    cfg = MultiQuadConfig()
    # Check default numerical values
    assert cfg.policy_freq == 250.0
    assert cfg.sim_steps_per_action == 1
    assert cfg.episode_length == 2048
    assert cfg.obs_noise == 0.0
    assert cfg.act_noise == 0.0
    assert cfg.max_thrust_range == 0.3
    assert cfg.num_quads == 2
    assert cfg.cable_length == jp.array(0.4) or cfg.cable_length == 0.4
    assert cfg.payload_mass == jp.array(0.01) or cfg.payload_mass == 0.01
    # Reward coefficients keys
    expected_keys = {"distance_reward_coef", "z_distance_reward_coef", "velocity_reward_coef", 
                     "safe_distance_coef", "up_reward_coef", "linvel_reward_coef", 
                     "ang_vel_reward_coef", "linvel_quad_reward_coef", "taut_reward_coef", 
                     "collision_penalty_coef", "out_of_bounds_penalty_coef", "smooth_action_coef", 
                     "action_energy_coef"}
    assert set(cfg.reward_coeffs.keys()) == expected_keys
