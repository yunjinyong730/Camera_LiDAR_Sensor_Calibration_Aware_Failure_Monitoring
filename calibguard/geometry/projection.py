"""Projection utilities for KITTI camera-LiDAR geometry."""

from __future__ import annotations

import numpy as np


def to_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    """Append a homogeneous 1 column to Nx3 points."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def project_lidar_to_image(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    T_velo_to_cam_4x4: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """Project Velodyne points into the left color camera image.

    Parameters
    ----------
    points_velo:
        Nx4 KITTI Velodyne points `[x, y, z, reflectance]`.
    P2:
        3x4 camera projection matrix.
    R0_rect:
        3x3 rectification matrix.
    T_velo_to_cam_4x4:
        4x4 LiDAR-to-camera transform.
    image_shape:
        Image shape used for filtering visible points.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Valid projected points `[u, v, depth, reflectance]`.
    """
    h, w = image_shape[:2]
    if points_velo.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    pts_h = to_homogeneous(points_velo[:, :3]).T  # 4 x N
    refl = points_velo[:, 3:4] if points_velo.shape[1] > 3 else np.zeros((points_velo.shape[0], 1), dtype=np.float32)

    R0 = np.eye(4, dtype=np.float32)
    R0[:3, :3] = R0_rect.astype(np.float32)

    cam = (P2.astype(np.float32) @ R0 @ T_velo_to_cam_4x4.astype(np.float32) @ pts_h).T
    depth = cam[:, 2]
    u = cam[:, 0] / np.maximum(depth, 1e-6)
    v = cam[:, 1] / np.maximum(depth, 1e-6)

    valid = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.concatenate([u[:, None], v[:, None], depth[:, None], refl], axis=1)
    return out[valid].astype(np.float32)


def mean_reprojection_error(proj_a: np.ndarray, proj_b: np.ndarray, max_points: int = 4096) -> float:
    """Compute a simple mean pixel distance between two corresponding projections.

    This assumes both projections were computed from the same LiDAR point ordering.
    After filtering, lengths may differ. For a stable lightweight metric we compare
    the first `min(len(a), len(b), max_points)` points.
    """
    n = min(len(proj_a), len(proj_b), max_points)
    if n == 0:
        return float("nan")
    diff = proj_a[:n, :2] - proj_b[:n, :2]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _project_raw_uvd(points_velo: np.ndarray, P2: np.ndarray, R0_rect: np.ndarray, T_velo_to_cam_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project all LiDAR points without image-boundary filtering.

    Returns raw `(u, v, depth)` arrays in the original point order. This helper is
    used for corresponding-point reprojection error, where comparing the same
    LiDAR index matters.
    """
    if points_velo.size == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    pts_h = to_homogeneous(points_velo[:, :3]).T
    R0 = np.eye(4, dtype=np.float32)
    R0[:3, :3] = R0_rect.astype(np.float32)
    cam = (P2.astype(np.float32) @ R0 @ T_velo_to_cam_4x4.astype(np.float32) @ pts_h).T
    depth = cam[:, 2]
    u = cam[:, 0] / np.maximum(depth, 1e-6)
    v = cam[:, 1] / np.maximum(depth, 1e-6)
    return u.astype(np.float32), v.astype(np.float32), depth.astype(np.float32)


def mean_corresponding_reprojection_error(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    T_a: np.ndarray,
    T_b: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    max_points: int = 4096,
) -> float:
    """Mean pixel distance between projections of the same LiDAR points.

    The previous lightweight metric compared the first N visible points after
    filtering. That can accidentally compare different original LiDAR points when
    drift moves points out of the image. This function fixes that by filtering on
    points visible under both transforms and then comparing by original index.
    """
    h, w = image_shape[:2]
    ua, va, za = _project_raw_uvd(points_velo, P2, R0_rect, T_a)
    ub, vb, zb = _project_raw_uvd(points_velo, P2, R0_rect, T_b)
    valid = (
        (za > 0.1) & (zb > 0.1) &
        (ua >= 0) & (ua < w) & (va >= 0) & (va < h) &
        (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
    )
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return float("nan")
    if idx.size > max_points:
        idx = idx[:max_points]
    diff = np.stack([ua[idx] - ub[idx], va[idx] - vb[idx]], axis=1)
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def project_lidar_to_image_indexed(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    T_velo_to_cam_4x4: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    """Project LiDAR points and keep the original LiDAR point index.

    Returns
    -------
    np.ndarray, shape (M, 5)
        Valid projected points `[u, v, depth, reflectance, original_index]`.

    Why this function matters
    -------------------------
    For calibration recovery visualization we must compare the *same* LiDAR
    points under normal/drifted/recovered transforms. If we only compare the
    first N visible points after filtering, points may no longer correspond when
    drift moves some points out of the image. The original index solves that.
    """
    h, w = image_shape[:2]
    if points_velo.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    u, v, depth = _project_raw_uvd(points_velo, P2, R0_rect, T_velo_to_cam_4x4)
    refl = points_velo[:, 3] if points_velo.shape[1] > 3 else np.zeros((points_velo.shape[0],), dtype=np.float32)
    valid = (depth > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(valid).astype(np.float32)
    return np.stack([u[valid], v[valid], depth[valid], refl[valid], idx], axis=1).astype(np.float32)


def corresponding_projected_points(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    transforms: list[np.ndarray],
    image_shape: tuple[int, int] | tuple[int, int, int],
    max_points: int = 2048,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return projections of identical LiDAR indices for multiple transforms.

    Parameters
    ----------
    transforms:
        List of transforms, e.g. `[T_gt, T_bad, T_rec]`.

    Returns
    -------
    indices:
        Original LiDAR point indices visible under every transform.
    uvds:
        List of arrays, one per transform. Each array has shape `(K, 3)` and
        contains `[u, v, depth]` for the same `indices`.
    """
    if not transforms:
        raise ValueError("transforms must not be empty")
    h, w = image_shape[:2]
    raws = [_project_raw_uvd(points_velo, P2, R0_rect, T) for T in transforms]
    valid = np.ones((points_velo.shape[0],), dtype=bool)
    for u, v, z in raws:
        valid &= (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return idx.astype(np.int32), [np.zeros((0, 3), dtype=np.float32) for _ in transforms]

    # Spread the sampled points over the visible set instead of taking the first
    # N points, which often over-represents a narrow scanline region.
    if idx.size > max_points:
        take = np.linspace(0, idx.size - 1, max_points).astype(np.int64)
        idx = idx[take]

    uvds = []
    for u, v, z in raws:
        uvds.append(np.stack([u[idx], v[idx], z[idx]], axis=1).astype(np.float32))
    return idx.astype(np.int32), uvds


def reprojection_recovery_summary(
    points_velo: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    T_gt: np.ndarray,
    T_bad: np.ndarray,
    T_rec: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> dict[str, float | str]:
    """Compute human-readable recovery metrics for a synthetic drift demo."""
    drift_err = mean_corresponding_reprojection_error(points_velo, P2, R0_rect, T_gt, T_bad, image_shape)
    rec_err = mean_corresponding_reprojection_error(points_velo, P2, R0_rect, T_gt, T_rec, image_shape)
    if not np.isfinite(drift_err) or drift_err <= 1e-6 or not np.isfinite(rec_err):
        rate = float("nan")
        status = "UNKNOWN"
    else:
        rate = 100.0 * (drift_err - rec_err) / drift_err
        if rate >= 30.0:
            status = "RECOVERED"
        elif rate >= 5.0:
            status = "PARTIALLY_RECOVERED"
        else:
            status = "NOT_RECOVERED"
    return {
        "drifted_reprojection_error_px": float(drift_err),
        "recovered_reprojection_error_px": float(rec_err),
        "recovery_rate_percent": float(rate),
        "recovery_status": status,
    }

