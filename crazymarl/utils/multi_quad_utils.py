import jax.numpy as jp
from jax import jit

@jit
def R_from_quat(q: jp.ndarray) -> jp.ndarray:
    """Compute a 3×3 rotation matrix from a quaternion [w, x, y, z]."""
    q = q / jp.linalg.norm(q)
    w, x, y, z = q
    r1 = jp.array([1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)])
    r2 = jp.array([2*(x*y + z*w),   1 - 2*(x*x + z*z), 2*(y*z - x*w)])
    r3 = jp.array([2*(x*z - y*w),   2*(y*z + x*w),     1 - 2*(x*x + y*y)])
    return jp.stack([r1, r2, r3])


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
    Compute the angle between each quad's local up-axis (3rd column of R) 
    and the global z-axis, in radians.
    
    Args:
      rots: Array of shape (num_quads*9,) or (num_quads, 9) representing
            flattened rotation matrices in row-major order.
      eps:  Small epsilon to guard against edge‐cases (optional).
    Returns:
      angles: Array of shape (num_quads,) with angles in [0, pi].
    """
    # Reshape into (num_quads, 3, 3)
    R = rots.reshape(-1, 3, 3)
    # Extract cos(theta) = R[2,2] (dot with [0,0,1])
    cos_theta = R[:, 2, 2]
    # Clip for numerical stability
    cos_theta = jp.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    # Return angle
    return jp.arccos(cos_theta)