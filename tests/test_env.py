import jax
import jax.numpy as jnp
import pytest
from crazymarl.envs.multi_quad_env import MultiQuadEnv
from crazymarl.configs.multi_quad_config import MultiQuadConfig

def test_env_reset_and_step():
    # Initialize env and PRNG
    cfg = MultiQuadConfig(num_quads=2, obs_noise=0.0, act_noise=0.0)
    env = MultiQuadEnv(**cfg.__dict__)
    key = jax.random.PRNGKey(42)
    # Reset env
    state = env.reset(key)
    # Check state attributes
    assert hasattr(state, 'pipeline_state')
    assert hasattr(state, 'obs')
    assert hasattr(state, 'reward')
    assert hasattr(state, 'done')
    assert hasattr(state, 'metrics')
    # obs shape should be 6 + num_quads*22
    expected_len = 6 + cfg.num_quads * 22
    assert state.obs.shape[0] == expected_len
    # Metrics contain time, reward, max_thrust
    for m in ('time', 'reward', 'max_thrust'):
        assert m in state.metrics
    # Step with zero action
    action = jnp.zeros(env.sys.nu)
    new_state = env.step(state, action)
    # Check that new_state is different type
    assert new_state.obs.shape == state.obs.shape
    assert isinstance(new_state.reward, jnp.ndarray)
    assert isinstance(new_state.done, jnp.ndarray)
    # reward should be finite scalar
    assert jnp.isfinite(new_state.reward)

@pytest.mark.parametrize("traj", [
    [[0.0,0.0,1.0], [1.0,1.0,2.0]],
    None
])
def test_env_trajectory(traj):
    # Test that trajectory updates target_position
    kwargs = {'num_quads':1, 'obs_noise':0.0, 'act_noise':0.0}
    if traj is not None:
        kwargs['trajectory'] = traj
    cfg = MultiQuadConfig(**kwargs)
    env = MultiQuadEnv(**cfg.__dict__)
    if traj is not None:
        # initial target_position should be first traj point
        assert jnp.allclose(env.target_position, jnp.array(traj[0], dtype=jnp.float32))
    else:
        # default goal_center
        assert jnp.allclose(env.target_position, env.goal_center)
