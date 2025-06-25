from .environments import (

    MultiQuad,

   
)



def make(env_id: str, episode_length: int = 1000, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
   
    elif env_id == "multiquad_ix4":
        env = MultiQuad(**env_kwargs,episode_length=episode_length)
        print(f"Using episode_length={episode_length} for {env_id}")


    
    return env

registered_envs = [
   
    "multiquad_ix4",
   
]
