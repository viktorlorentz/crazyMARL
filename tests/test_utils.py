import pytest
import jax.numpy as jp
import numpy as np
from crazymarl.utils.multi_quad_utils import R_from_quat, angle_between, upright_angles


def is_close(a, b, tol=1e-6):
    return np.allclose(np.array(a), np.array(b), atol=tol)


def test_R_from_quat_identity():
    """
    Test that the identity quaternion [1,0,0,0] yields the identity rotation matrix.
    """
    q = jp.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    R = R_from_quat(q)  # shape (3,3)
    expected = jp.eye(3)
    assert R.shape == (3, 3)
    assert is_close(R, expected)


def test_R_from_quat_random():
    """
    Test random normalized quaternion yields an orthonormal rotation matrix.
    """
    # random unit quaternion
    q = jp.array([0.5, 0.5, 0.5, 0.5])
    R = R_from_quat(q)
    # Check orthonormal: R.T @ R = I
    I = jp.dot(R.T, R)
    assert is_close(I, jp.eye(3))


def test_angle_between_basic():
    """
    Test angle_between for known cases: identical, orthogonal, opposite.
    """
    v1 = jp.array([1.0, 0.0, 0.0])
    v2 = jp.array([1.0, 0.0, 0.0])
    assert pytest.approx(0.0, abs=1e-2) == angle_between(v1, v2)

    v2 = jp.array([0.0, 1.0, 0.0])
    assert pytest.approx(jp.pi/2, abs=1e-2) == angle_between(v1, v2)

    v2 = jp.array([-1.0, 0.0, 0.0])
    assert pytest.approx(jp.pi, abs=1e-2) == angle_between(v1, v2)


def test_upright_angles_from_matrix():
    """
    Test upright_angles for rotation matrices: identity (angle=acos(1)=0) and 90-degree tilt.
    """
    # Identity rotation
    R_id = jp.eye(3)
    angles = upright_angles(R_id)
    assert angles.shape == () or angles.shape == (1,)
    assert pytest.approx(0.0, abs=1e-2) == angles

    # 90-degree rotation around x-axis tilts up-vector to y-axis
    # so angle between up-axis and global z-axis is 90 degrees
    # rotation matrix for 90 deg around x: rots[2,2] = cos(90deg) = 0
    R_x90 = jp.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=jp.float32)
    angle = upright_angles(R_x90)
    assert pytest.approx(jp.pi/2, abs=1e-2) == angle
