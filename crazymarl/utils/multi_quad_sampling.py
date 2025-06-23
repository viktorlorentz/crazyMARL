import jax
from jax import numpy as jp


def generate_configuration(
    key: jax.Array,
    num_quads: int,
    cable_length: float,
    target_position: jp.ndarray,
    target_start_ratio: float
):
    """Sample a single payload+quad configuration."""
    subkeys = jax.random.split(key, 5)
    min_qz, min_pz = 0.008, 0.0055

    # payload sample
    xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    pz = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    payload = jp.array([xy[0], xy[1], pz.clip(min_pz, 3.0)])
    is_target_start = jax.random.uniform(subkeys[1]) < target_start_ratio
    payload = jp.where(
        is_target_start,
        target_position + jax.random.normal(subkeys[1], (3,)) * 0.02,
        payload
    )

    # spherical sampling for quads
    mean_r, std_r = cable_length, cable_length / 3
    clip_r = (0.05, cable_length)
    mean_th, std_th = jp.pi / 4, jp.pi / 8
    std_phi = jp.pi / (num_quads + 1)

    r = jp.clip(
        mean_r + std_r * jax.random.normal(subkeys[2], (num_quads,)),
        *clip_r
    )
    th = mean_th + std_th * jax.random.normal(subkeys[3], (num_quads,))
    offset = jax.random.uniform(subkeys[4], (), minval=-jp.pi, maxval=jp.pi)
    phi = (
        jp.arange(num_quads) * (2 * jp.pi / num_quads)
        + std_phi * jax.random.normal(subkeys[4], (num_quads,))
        + offset
    )

    x = r * jp.sin(th) * jp.cos(phi) + payload[0]
    y = r * jp.sin(th) * jp.sin(phi) + payload[1]
    z = jp.clip(r * jp.cos(th) + payload[2], min_qz, 3.0)
    quads = jp.stack([x, y, z], axis=1)

    return payload, quads


def generate_filtered_configuration_batch(
    key: jax.Array,
    batch_size: int,
    num_quads: int,
    cable_length: float,
    target_position: jp.ndarray,
    target_start_ratio: float
):
    """Oversample and filter configurations to enforce min distances."""
    os_factor = round(num_quads / 2) + 1
    M = os_factor * batch_size
    keys = jax.random.split(key, M)

    payloads, quadss = jax.vmap(
        generate_configuration,
        in_axes=(0, None, None, None, None)
    )(keys, num_quads, cable_length, target_position, target_start_ratio)

    # quad-quad distances
    diffs = quadss[:, :, None, :] - quadss[:, None, :, :]
    dists = jp.linalg.norm(diffs, axis=-1)
    eye = jp.eye(num_quads, dtype=bool)[None]
    dists = jp.where(eye, jp.inf, dists)
    min_quad = jp.min(dists, axis=(1, 2))

    # quad-payload distances
    pd = quadss - payloads[:, None, :]
    min_payload = jp.min(jp.linalg.norm(pd, axis=-1), axis=1)

    # mask and select
    mask = (min_quad >= 0.16) & (min_payload >= 0.07)
    idx = jp.argsort(-mask.astype(jp.int32))[:batch_size]

    return payloads[idx], quadss[idx]
