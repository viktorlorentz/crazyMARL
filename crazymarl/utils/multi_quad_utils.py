import jax.numpy as jp


def R_from_quat(q: jp.ndarray) -> jp.ndarray:
    """Compute a 3×3 rotation matrix from a quaternion [w, x, y, z]."""
    q = q / jp.linalg.norm(q)
    w, x, y, z = q
    r1 = jp.array([1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)])
    r2 = jp.array([2*(x*y + z*w),   1 - 2*(x*x + z*z), 2*(y*z - x*w)])
    r3 = jp.array([2*(x*z - y*w),   2*(y*z + x*w),     1 - 2*(x*x + y*y)])
    return jp.stack([r1, r2, r3])


def angle_between(v1: jp.ndarray, v2: jp.ndarray) -> jp.ndarray:
    """Return the angle between two vectors."""
    norm1 = jp.linalg.norm(v1)
    norm2 = jp.linalg.norm(v2)
    dot = jp.dot(v1, v2)
    cos_theta = dot / (norm1 * norm2 + 1e-6)
    cos_theta = jp.clip(cos_theta, -1.0, 1.0)
    return jp.arccos(cos_theta)
