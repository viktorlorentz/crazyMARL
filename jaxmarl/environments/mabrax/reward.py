import jax.numpy as jp


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

    # payload tracking rewards
    team_obs = obs[:6]
    payload_err = team_obs[:3]
    payload_linlv = team_obs[3:6]
    dis = jp.linalg.norm(payload_err)
    tracking_reward = cfg.reward_coeffs["distance_reward_coef"] * er(dis)

    # per-quad observations
    quad_obs = [obs[6 + i*24 : 6 + (i+1)*24] for i in range(cfg.num_quads)]
    rels = jp.stack([q[:3] for q in quad_obs])
    rots = jp.stack([q[3:12] for q in quad_obs])
    linvels = jp.stack([q[12:15] for q in quad_obs])
    angvels = jp.stack([q[15:18] for q in quad_obs])

    # safe-distance reward
    if cfg.num_quads > 1:
        d = jp.linalg.norm(rels[:, None, :] - rels[None, :, :], axis=-1)
        eye = jp.eye(cfg.num_quads, dtype=bool)
        pairwise = jp.where(eye, jp.inf, d)
        safe_distance = jp.mean(jp.clip(3*(pairwise - 0.18)/0.02, -20, 1))
    else:
        safe_distance = 1.0

    # upright reward
    up_reward = jp.mean(er(angles))

    # taut-string reward
    quad_dists = jp.linalg.norm(rels, axis=-1)
    quad_heights = rels[:, 2]
    taut_reward = (jp.sum(quad_dists) + jp.sum(quad_heights)) / cfg.cable_length

    # angular & linear velocity rewards
    ang_vel_vals = jp.stack([er(jp.linalg.norm(jvp, axis=-1)) for jvp in angvels])
    ang_vel_reward = (0.5 + 3*er(dis, 20)) * jp.mean(ang_vel_vals)
    linvel_vals = jp.stack([er(jp.linalg.norm(jvp, axis=-1)) for jvp in linvels])
    linvel_quad_reward = (0.5 + 6*er(dis, 20)) * jp.mean(linvel_vals)

    # penalties & energy cost
    collision_penalty = cfg.reward_coeffs["collision_penalty_coef"] * collision
    oob_penalty = cfg.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds
    smooth_penalty = cfg.reward_coeffs["smooth_action_coef"] * jp.mean(jp.abs(action - last_action))**2
    thrust_cmds = 0.5 * (action + 1.0)
    thrust_extremes = jp.exp(-50*jp.abs(thrust_cmds)) + jp.exp(50*(thrust_cmds - 1))
    thrust_extremes = jp.where(jp.abs(action) > 1.0, 1.0 + 0.1*jp.abs(action), thrust_extremes)
    energy_penalty = cfg.reward_coeffs["action_energy_coef"] * jp.mean(thrust_extremes)

    # combine stability & safety
    stability = (
        cfg.reward_coeffs["up_reward_coef"] * up_reward +
        cfg.reward_coeffs["taut_reward_coef"] * taut_reward +
        cfg.reward_coeffs["ang_vel_reward_coef"] * ang_vel_reward +
        cfg.reward_coeffs["linvel_quad_reward_coef"] * linvel_quad_reward
    )
    safety = (
        cfg.reward_coeffs["safe_distance_coef"] * safe_distance +
        collision_penalty + oob_penalty + smooth_penalty + energy_penalty
    )

    return tracking_reward * (stability + safety)
