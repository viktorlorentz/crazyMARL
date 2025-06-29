import jax.numpy as jp
from crazymarl.observations.multi_quad_observation import parse_obs

from crazymarl.utils.multi_quad_utils import upright_angles


def calc_reward(
    obs: jp.ndarray,
    sim_time: float,
    collision: bool,
    out_of_bounds: bool,
    action: jp.ndarray,
    angles: jp.ndarray,
    last_action: jp.ndarray,
    target_position: jp.ndarray,
    data,
    max_thrust: float,
    cfg
) -> jp.ndarray:
    """
    Compute the scalar reward combining tracking, stability, safety, and penalties.
    """
    er = lambda x, s=2: jp.exp(-s * jp.abs(x))

    # extract obs components
    payload_err, payload_linlv, rels, rots, linvels, angvels, prev_last_actions = parse_obs(obs, cfg.num_quads)
    dis = jp.linalg.norm(payload_err)
    tracking_reward = cfg.reward_coeffs["distance_reward_coef"] * er(dis)

    # Velocity alignment.
    target_dir  = payload_err / (dis + 1e-6)
    vel = jp.linalg.norm(payload_linlv)
    # Avoid division by zero. 
    vel_dir = jp.where(jp.abs(vel) > 1e-6, payload_linlv / vel, jp.zeros_like(payload_linlv))
  

    aligned_vel = er(1 - jp.dot(vel_dir, target_dir), jp.minimum(40 * dis, 4)) # dotprod = 1  => vel is perfectly aligned
    tracking_reward += aligned_vel

    tracking_reward /=2 # normalize tracking reward to be in [0, 1]



    # safe-distance reward (mean over all pairs)
    if cfg.num_quads > 1:
        d = jp.linalg.norm(rels[:, None, :] - rels[None, :, :], axis=-1)
        eye = jp.eye(cfg.num_quads, dtype=bool)
        pairwise = jp.where(eye, jp.inf, d)
        # make sure the distance is positive
        pair_distance = jp.abs(pairwise) - 0.15
        safe_distance = jp.mean(jp.clip(40 * pair_distance, 0, 1)) # encourage more than dist of 0.15m between quads and full reward for distances above 0.175m

    else:
        safe_distance = 1.0

    # from rots get the up vector and compute the upright reward
    angles = upright_angles(rots)
    # upright reward = mean over all quads
    up_reward = jp.mean(er(angles))

    # taut-string reward = sum of distances + heights
    quad_dists   = jp.linalg.norm(rels, axis=-1)
    quad_heights = rels[:, 2]
    taut_reward  = (jp.mean(quad_dists) + 5 * jp.mean(quad_heights)) / cfg.cable_length

    # angular & linear velocity
    vel_shaping = jp.maximum(4.0 * er(dis, 100.0), 0.01 ) # low velocity tolerance close to the target

    
 

    yaw_vels = jp.linalg.norm(angvels[:, 2]) # only yaw velocity matters
    ang_vel_reward      = jp.mean(er(yaw_vels))

    # This function computes the velocity reward based on the distance, current and maximum velocities. 
    # Close to the target it only alows low velocity and further in allows up to max_vel
    vel_reward_function = lambda vel, max_vel: jp.exp(-((vel / (jp.minimum(3 * dis, 1) + 0.02)) / max_vel) ** 8)

    linvel_quad_reward  = jp.mean(vel_reward_function(jp.linalg.norm(linvels, axis=-1), 2.0)) #quad max 2m/s
    payload_velocity_reward = vel_reward_function(jp.linalg.norm(payload_linlv), 1.5) #payload max 1.5m/s


    # penalties
    collision_penalty = cfg.reward_coeffs["collision_penalty_coef"] * collision
    oob_penalty       = cfg.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds

    # action differences
    action_diff= jp.abs(action - last_action)
    thrust_deviations= jp.abs(action - jp.mean(action, axis=0)) # mean deviation from the mean action
    smooth_penalty  = cfg.reward_coeffs["smooth_action_coef"] * (jp.mean(action_diff) + jp.mean(thrust_deviations) )
    thrust_cmds = 0.5 * (action + 1.0)
    thrust_extremes = jp.exp(-50 * jp.abs(thrust_cmds)) + jp.exp(50 * (thrust_cmds - 1)) # 1 if thrust_cmds is 0 or 1 and going to 0 in the middle
    # if actions out of bounds lead them to action space
    thrust_extremes = jp.where(jp.abs(action)> 1.0, 1.0 + 0.1*jp.abs(action), thrust_extremes)  

    energy_penalty    = cfg.reward_coeffs["action_energy_coef"] * jp.mean(thrust_extremes)


    stability = (cfg.reward_coeffs["up_reward_coef"] * up_reward
                 + cfg.reward_coeffs["taut_reward_coef"] * taut_reward
                 + cfg.reward_coeffs["ang_vel_reward_coef"] * ang_vel_reward
                 + cfg.reward_coeffs["linvel_quad_reward_coef"] * linvel_quad_reward
                 + cfg.reward_coeffs["linvel_reward_coef"] * payload_velocity_reward)
                 
    stability /= 5.0  # normalize stability to be in [0, 1]
    
    safety = safe_distance * cfg.reward_coeffs["safe_distance_coef"] \
           + collision_penalty + oob_penalty + smooth_penalty + energy_penalty
    
    safety /= 5.0

    reward = tracking_reward * stability + safety
  
    return reward
