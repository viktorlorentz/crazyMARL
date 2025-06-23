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

    # payload tracking
    payload_err = obs[:3]
    dis = jp.linalg.norm(payload_err)
    track_r = cfg.reward_coeffs["distance_reward_coef"] * er(dis)

    # always build per-quad rels for later use
    rels = jp.stack([obs[6 + i*24 : 6 + i*24 +3] for i in range(cfg.num_quads)])

    # safe-distance
    if cfg.num_quads > 1:
        d = jp.linalg.norm(rels[:, None, :] - rels[None, :, :], axis=-1)
        eye = jp.eye(cfg.num_quads, dtype=bool)
        pairwise = jp.where(eye, jp.inf, d)
        safe_dist = jp.mean(jp.clip(3*(pairwise - 0.18)/0.02, -20, 1))
    else:
        safe_dist = 1.0

    up_r = jp.mean(er(angles))

    quad_dists = jp.linalg.norm(rels, axis=-1)
    quad_heights = rels[:, 2]
    taut_r = (jp.sum(quad_dists) + jp.sum(quad_heights)) / cfg.cable_length

    # vectorize ang_r & linquad_r
    ang_vals = jp.stack([
        er(jp.linalg.norm(obs[6 + i*24 + 15 : 6 + i*24 + 18]))
        for i in range(cfg.num_quads)
    ])
    ang_r = (0.5 + 3*er(dis, 20)) * jp.mean(ang_vals)

    lin_vals = jp.stack([
        er(jp.linalg.norm(obs[6 + i*24 + 12 : 6 + i*24 + 15]))
        for i in range(cfg.num_quads)
    ])
    linquad_r = (0.5 + 6*er(dis, 20)) * jp.mean(lin_vals)

    penalties = (
        cfg.reward_coeffs["collision_penalty_coef"] * collision +
        cfg.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds +
        cfg.reward_coeffs["smooth_action_coef"] * jp.mean(jp.abs(action - last_action))**2
    )

    thrust_cmds = 0.5*(action + 1.0)
    extremes = jp.exp(-50*jp.abs(thrust_cmds)) + jp.exp(50*(thrust_cmds - 1))
    energy_p = cfg.reward_coeffs["action_energy_coef"] * jp.mean(extremes)

    stability = (
        cfg.reward_coeffs["up_reward_coef"] * up_r +
        cfg.reward_coeffs["taut_reward_coef"] * taut_r +
        cfg.reward_coeffs["ang_vel_reward_coef"] * ang_r +
        cfg.reward_coeffs["linvel_quad_reward_coef"] * linquad_r
    )

    safety = (
        cfg.reward_coeffs["safe_distance_coef"] * safe_dist + penalties + energy_p
    )

    return track_r * (stability + safety)
