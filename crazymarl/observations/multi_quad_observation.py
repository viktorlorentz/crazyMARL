import jax
from jax import numpy as jp
from brax import base
from crazymarl.utils.multi_quad_utils import R_from_quat


def build_obs(
    data: base.State,
    last_action: jp.ndarray,
    target_position: jp.ndarray,
    obs_noise: float,
    noise_key: jax.Array,
    ids: dict,
) -> jp.ndarray:
    """
    Construct the observation vector (payload + per-quad states) with optional noise.
    """
    payload_pos = data.xpos[ids["payload_body_id"]]
    payload_linvel = data.cvel[ids["payload_body_id"]][3:6]
    payload_error = target_position - payload_pos
    dist = jp.linalg.norm(payload_error)
    payload_error = payload_error / jp.maximum(dist, 1.0)

    team_obs= 6
    quad_obs = 18 + 4 # 3 rel-pos, 9 rotation, 3 linvel, 3 angvel, 4 last_action
    num_quads = len(ids["quad_body_ids"])
    num_actions = num_quads * 4

    obs_length = team_obs + num_quads * quad_obs + num_actions

    quad_obs_starts = team_obs + jp.arange(num_quads) * quad_obs

    obs = jp.zeros(obs_length)

    #first the team observations
    obs[0:2] = payload_error
    obs[2:5] = payload_linvel
    # Now the individual quad observations
    obs[quad_obs_starts+0:quad_obs_starts+3] = data.xpos[ids["quad_body_ids"]] - payload_pos # relative position
    obs[quad_obs_starts+3:quad_obs_starts+12] = R_from_quat(data.xquat[ids["quad_body_ids"]]).ravel() # rotation as  flat matrix
    obs[quad_obs_starts+12:quad_obs_starts+15] = data.cvel[ids["quad_body_ids"]][:, 3:6] # linear velocity
    obs[quad_obs_starts+15:quad_obs_starts+18] = data.cvel[ids["quad_body_ids"]][:, :3] # angular velocity

    # Now the last action of each quad
    obs[quad_obs_starts+18:quad_obs_starts+22] = jp.clip(last_action[jp.arange(num_quads):jp.arange(num_quads)+4], -1.0, 1.0)



    # if obs_noise > 0.0:
    #     noises = [jp.ones(3)*0.005, jp.ones(3)*0.05]
    #     for _ in range(len(ids["quad_body_ids"])):
    #         noises += [
    #             jp.ones(3)*0.02, jp.ones(9)*0.005,
    #             jp.ones(3)*0.05, jp.ones(3)*0.08,
    #             jp.ones(3)*0.05, jp.ones(3)*0.05
    #         ]
    #     noises.append(jp.zeros_like(last_action))
    #     noise_lookup = jp.concatenate(noises)
    #     noise = obs_noise * noise_lookup * jax.random.normal(noise_key, obs.shape)
    #     obs = obs + noise

    return obs




def get_ix4_mappings(num_quads: int):
    """
    Generate dynamic observation and action mappings for multiquad_ix4 environments.

    Args:
        num_quads: Number of quadrotors in the environment.

    Returns:
        Tuple of (action_mapping, observation_mapping) where
        - action_mapping maps agent names to JAX arrays of action indices,
        - observation_mapping maps agent names to JAX arrays of observation indices.
    """
    # build dynamic observation ranges
    dyn_ix = {}

    team_obs= 6
    quad_obs = 18 + 4 # 3 rel-pos, 9 rotation, 3 linvel, 3 angvel, 4 last_action
    num_actions = num_quads * 4


    for i in range(num_quads):
        agent = []
        agent += [0,1,2]  # payload_error
        agent += [3,4,5]  # payload_linvel
        
        quad_obs_indices = jp.arange(team_obs + i * quad_obs, team_obs + (i + 1) * quad_obs - 1)
        agent += quad_obs_indices.tolist()  # own full state block

        # ids of othergents without own
        for j in range(num_quads):
            if j != i:
                rel_pos_start = team_obs + j * quad_obs
                agent += [rel_pos_start, rel_pos_start + 1, rel_pos_start + 2]

    

        dyn_ix[f"agent_{i}"] = agent

    # build action mapping
    action_map = {
        f"agent_{i}": jp.arange(i * 4, (i + 1) * 4)
        for i in range(num_quads)
    }
    # build observation mapping
    obs_map = {agent: jp.array(ids) for agent, ids in dyn_ix.items()}

    print("obs_map", obs_map)
    return action_map, obs_map