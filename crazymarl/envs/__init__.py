import brax.envs
from .multi_quad_env import MultiQuadEnv

brax.envs.register_environment('multiquad', MultiQuadEnv)
