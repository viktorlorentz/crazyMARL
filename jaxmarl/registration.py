from .environments import (

    Ant,
    Humanoid,
    Hopper,
    Walker2d,
    HalfCheetah,
    MultiQuad,
    Quad,
   
)



def make(env_id: str, episode_length: int = 1000, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

   
    if env_id == "ant_4x2":
        env = Ant(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah(**env_kwargs)
    elif env_id == "hopper_3x1":
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env = Walker2d(**env_kwargs)
    elif env_id == "multiquad_2x4":
        env = MultiQuad(**env_kwargs)
    elif env_id == "multiquad_ix4":
        env = MultiQuad(**env_kwargs,episode_length=episode_length)
        print(f"Using episode_length={episode_length} for {env_id}")
    elif env_id == "quad_1x4":
        env = Quad(**env_kwargs)

    
    return env

registered_envs = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    "multiquad_2x4",
    "multiquad_ix4",
    "quad_1x4",
    "storm",
    "storm_2p",
    "storm_np",
    "hanabi",
    "overcooked",
    "overcooked_v2",
    "coin_game",
    "jaxnav",
]
