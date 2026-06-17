"""Small SE(3) utilities for camera-LiDAR calibration.

The project represents calibration drift/correction as a 6D vector:

    [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m]

Rotation order is Rz(yaw) @ Ry(pitch) @ Rx(roll). This convention is explicitly
written here so future experiments remain reproducible.
"""

from __future__ import annotations

import math
import numpy as np


def deg2rad(x: float | np.ndarray) -> float | np.ndarray:
    """Convert degrees to radians."""
    return x * math.pi / 180.0


def euler_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll, pitch, yaw in degrees.

    Parameters
    ----------
    roll_deg, pitch_deg, yaw_deg:
        Euler angles in degrees.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    r, p, y = deg2rad(roll_deg), deg2rad(pitch_deg), deg2rad(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def vec6_to_transform(vec6: np.ndarray) -> np.ndarray:
    """Convert a 6D drift/correction vector to a 4x4 transform.

    Parameters
    ----------
    vec6:
        [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m]

    Returns
    -------
    np.ndarray, shape (4, 4)
        SE(3) transform matrix.
    """
    roll, pitch, yaw, tx, ty, tz = [float(v) for v in vec6]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_to_R(roll, pitch, yaw)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


def make_4x4(T_3x4: np.ndarray) -> np.ndarray:
    """Convert a KITTI 3x4 extrinsic matrix into a 4x4 homogeneous matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :4] = T_3x4.astype(np.float32)
    return T


def apply_delta_to_extrinsic(T_gt_3x4_or_4x4: np.ndarray, delta_vec6: np.ndarray) -> np.ndarray:
    """Apply a synthetic SE(3) drift/correction in camera coordinates.

    We use:

        T_new = Delta(delta_vec6) @ T_gt

    This follows the common setup where the LiDAR-to-camera extrinsic is perturbed
    by a small transform in the camera frame.
    """
    if T_gt_3x4_or_4x4.shape == (3, 4):
        T_gt = make_4x4(T_gt_3x4_or_4x4)
    elif T_gt_3x4_or_4x4.shape == (4, 4):
        T_gt = T_gt_3x4_or_4x4.astype(np.float32)
    else:
        raise ValueError(f"Expected 3x4 or 4x4 extrinsic, got {T_gt_3x4_or_4x4.shape}")
    return (vec6_to_transform(delta_vec6) @ T_gt).astype(np.float32)


def transform_to_vec6(T: np.ndarray) -> np.ndarray:
    """Convert a small SE(3) transform matrix to `[roll, pitch, yaw, tx, ty, tz]`.

    This is used to compute the exact inverse correction label from a synthetic
    drift transform. The Euler convention matches `euler_to_R`: `Rz @ Ry @ Rx`.
    The project samples only small drifts, so the non-gimbal branch is stable.
    """
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    # For R = Rz(yaw) @ Ry(pitch) @ Rx(roll):
    # R[2, 0] = -sin(pitch).
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < 1e-8:
        # Gimbal fallback. This should not occur for the small synthetic drifts
        # used in this project, but keeps the function numerically safe.
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    return np.array([
        roll * 180.0 / math.pi,
        pitch * 180.0 / math.pi,
        yaw * 180.0 / math.pi,
        t[0], t[1], t[2],
    ], dtype=np.float32)


def inverse_vec6(vec6: np.ndarray) -> np.ndarray:
    """Return the exact inverse transform represented as a 6D vector.

    Earlier versions of this project used `-drift` as the target correction. That
    is only an approximation. For a calibration model, the cleaner target is the
    inverse SE(3) transform, especially when translation and rotation are mixed.
    """
    T = vec6_to_transform(vec6)
    return transform_to_vec6(np.linalg.inv(T).astype(np.float32))


def normalize_vec6(vec6: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalize a 6D vector by fixed drift range scales."""
    return (vec6.astype(np.float32) / scale.astype(np.float32)).astype(np.float32)


def denormalize_vec6(vec6_norm: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Convert normalized model output back to degrees/meters."""
    return (vec6_norm.astype(np.float32) * scale.astype(np.float32)).astype(np.float32)
