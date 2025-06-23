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

    obs_list = [payload_error, payload_linvel]
    num_q = len(ids["quad_body_ids"])
    for i in range(num_q):
        bid = ids["quad_body_ids"][i]
        pos = data.xpos[bid]
        quat = data.xquat[bid]
        rel = pos - payload_pos
        rot = R_from_quat(quat).ravel()
        linvel = data.cvel[bid][3:6]
        angvel = data.cvel[bid][:3]
        linear_acc = data.cacc[bid][3:6]
        angular_acc= data.cacc[bid][:3]
        obs_list += [rel, rot, linvel, angvel, linear_acc, angular_acc]

    obs_list.append(jp.clip(last_action, -1.0, 1.0))
    obs = jp.concatenate(obs_list)

    if obs_noise > 0.0:
        noises = [jp.ones(3)*0.005, jp.ones(3)*0.05]
        for _ in range(len(ids["quad_body_ids"])):
            noises += [
                jp.ones(3)*0.02, jp.ones(9)*0.005,
                jp.ones(3)*0.05, jp.ones(3)*0.08,
                jp.ones(3)*0.05, jp.ones(3)*0.05
            ]
        noises.append(jp.zeros_like(last_action))
        noise_lookup = jp.concatenate(noises)
        noise = obs_noise * noise_lookup * jax.random.normal(noise_key, obs.shape)
        obs = obs + noise

    return obs
