import jax.numpy as jp
from jax import jit

@jit
def R_from_quat(q: jp.ndarray) -> jp.ndarray:
    """Compute rotation matrix(ices) from quaternion(s) [w, x, y, z].
       Supports q shape (4,) or (...,4) and returns (...,3,3)."""
    # normalize over last dim
    q = q / jp.linalg.norm(q, axis=-1, keepdims=True)
    w = q[..., 0]; x = q[..., 1]; y = q[..., 2]; z = q[..., 3]
    r1 = jp.stack([
        1 - 2*(y*y + z*z),
        2*(x*y - z*w),
        2*(x*z + y*w)
    ], axis=-1)
    r2 = jp.stack([
        2*(x*y + z*w),
        1 - 2*(x*x + z*z),
        2*(y*z - x*w)
    ], axis=-1)
    r3 = jp.stack([
        2*(x*z - y*w),
        2*(y*z + x*w),
        1 - 2*(x*x + y*y)
    ], axis=-1)
    # stack rows into matrix, preserving batch dims
    return jp.stack([r1, r2, r3], axis=-2)


@jit
def angle_between(v1: jp.ndarray, v2: jp.ndarray, eps: float = 1e-6) -> jp.ndarray:
    """Return the angle between two vectors.
    angle = arccos((v1 . v2) / (||v1|| * ||v2||))
    Args:
        v1: First vector.
        v2: Second vector.
        eps: Small value to avoid division by zero.
    Returns:
        Angle in radians between the two vectors.
    """
    # squared norms
    a = jp.sum(v1 * v1, axis=-1)
    b = jp.sum(v2 * v2, axis=-1)
    # dot product
    dot = jp.sum(v1 * v2, axis=-1)
    # single sqrt of product rather than two separate norms
    cos_theta = dot / jp.sqrt(a * b + eps)
    # guard against numerical drift
    cos_theta = jp.clip(cos_theta, -1.0, 1.0)
    return jp.arccos(cos_theta)


@jit

def upright_angles(rots: jp.ndarray, eps: float = 1e-6) -> jp.ndarray:
    """
    Compute the angle between each quad's local up-axis and global z-axis.
    Supports rots of shape (...,9) or (...,3,3).
    """
    # grab R[...,2,2] without any reshape
    if rots.shape[-1] == 9:
        cos_theta = rots[..., 8]
    elif rots.ndim >= 2 and rots.shape[-2:] == (3, 3):
        cos_theta = rots[..., 2, 2]
    else:
        raise ValueError(f"Invalid rots shape {rots.shape}, expected (...,9) or (...,3,3).")

    # clip for numerical stability and return
    cos_theta = jp.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    return jp.arccos(cos_theta)