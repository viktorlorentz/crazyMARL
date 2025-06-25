import jax
import jax.numpy as jnp
from jax import vmap, lax
from brax import base
from crazymarl.utils.multi_quad_utils import R_from_quat


@jax.jit
def build_obs(
    data: base.State,
    last_action: jnp.ndarray,       # shape (num_quads, 4)
    target_position: jnp.ndarray,   # shape (3,)
    obs_noise: float,
    noise_key: jax.Array,
    ids: dict,
) -> jnp.ndarray:
    # --- payload part ---
    payload_pos = data.xpos[ids["payload_body_id"]]                   # (3,)
    payload_linvel = data.cvel[ids["payload_body_id"], 3:6]           # (3,)
    err = target_position - payload_pos                              # (3,)
    dist = jnp.linalg.norm(err)
    payload_error = err / jnp.maximum(dist, 1.0)                      # (3,)
    # this make sure the payload error is capped to a unit vector

    # --- per-quad quantities (vectorized) ---
    quad_ids = jnp.array(ids["quad_body_ids"], dtype=int)             # (Q,)
    num_quads = quad_ids.shape[0]

    # positions
    quad_pos = data.xpos[quad_ids]                                   # (Q,3)
    rel_pos = quad_pos - payload_pos                                  # (Q,3)

    # rotations
    quats = data.xquat[quad_ids]                                      # (Q,4)
    rots = vmap(R_from_quat)(quats)                                   # (Q,3,3)
    rots_flat = rots.reshape(num_quads, -1)                           # (Q,9)

    # velocities
    linvels = data.cvel[quad_ids, 3:6]                                # (Q,3)
    angvels = data.cvel[quad_ids, 0:3]                                # (Q,3)

    # last action, clipped
    clipped_actions = last_action.reshape((num_quads, 4))
    clipped_actions = jnp.clip(clipped_actions, -1.0, 1.0)            # (Q,4)

    # stack each quad’s obs: [rel_pos, rot9, lin3, ang3, act4] -> (Q,22)
    per_quad = jnp.concatenate(
        [rel_pos, rots_flat, linvels, angvels, clipped_actions], axis=1
    )

    # flatten payload + quads
    flat_quads = per_quad.reshape(-1)                                 # (Q*22,)


    base_obs = jnp.concatenate([payload_error, payload_linvel], axis=0)  # (6,)

    obs = jnp.concatenate([base_obs, flat_quads], axis=0) # (6 + Q*22,)

    # --- optional noise ---
    # def add_noise(o):
    #     # build a lookup scale vector once
    #     payload_scale = jnp.concatenate([jnp.ones(3)*0.005, jnp.ones(3)*0.05])
    #     per_quad_scale = jnp.concatenate([
    #         jnp.ones(3)*0.02,  # rel pos
    #         jnp.ones(9)*0.005, # rot
    #         jnp.ones(3)*0.05,  # linvel
    #         jnp.ones(3)*0.08,  # angvel
    #         jnp.ones(4)*0.05   # actions
    #     ])  # (22,)
    #     noise_per_quad = jnp.tile(per_quad_scale, (num_quads,))  # (Q*22,)
    #     tail_scale = jnp.zeros(num_quads*4)                     # no noise on tail
    #     lookup = jnp.concatenate([payload_scale, noise_per_quad, tail_scale], axis=0)
    #     n = jax.random.normal(noise_key, o.shape)
    #     return o + obs_noise * lookup * n

    # obs = lax.cond(obs_noise > 0.0, add_noise, lambda o: o, obs)
    return obs


def get_ix4_mappings(num_quads: int):
    """
    Build per-agent action & observation index maps for a multi-quad (ix4) env.
    """
    team_obs = 6
    quad_obs = 22  # 3 + 9 + 3 + 3 + 4
    dyn_ix = {}

    for i in range(num_quads):
        agent = f"agent_{i}"
        idxs = []
        # payload_error & payload_linvel
        idxs += list(range(0, 3))
        idxs += list(range(3, 6))
        # own full quad block
        start = team_obs + i * quad_obs
        idxs += list(range(start, start + quad_obs))
        # just the relative-pos of the others
        for j in range(num_quads):
            if j != i:
                r0 = team_obs + j * quad_obs
                idxs += [r0, r0 + 1, r0 + 2]
        dyn_ix[agent] = jnp.array(idxs, dtype=int)

    action_map = {
        f"agent_{i}": jnp.arange(i * 4, (i + 1) * 4, dtype=int)
        for i in range(num_quads)
    }

    return action_map, dyn_ix


def parse_obs(obs: jnp.ndarray, num_quads: int):
    """
    Split a flat obs vector into:
      payload_error (3,),
      payload_linvel (3,),
      rel_pos (Q,3),
      rot matrices (Q,3,3),
      linvels (Q,3),
      angvels (Q,3),
      actions (Q,4)
    """
    team_obs = 6
    quad_obs = 22

    # payload
    payload_error = obs[0:3]
    payload_linvel = obs[3:6]

    # per-quad block
    flat_quads = obs[team_obs:]
    per_quads = flat_quads.reshape(num_quads, quad_obs)

    rel_pos    = per_quads[:,  0:3]
    rot_flat   = per_quads[:,  3:12]
    rots       = rot_flat.reshape(num_quads, 3, 3)
    linvels    = per_quads[:, 12:15]
    angvels    = per_quads[:, 15:18]
    actions    = per_quads[:, 18:22]

    return payload_error, payload_linvel, rel_pos, rots, linvels, angvels, actions