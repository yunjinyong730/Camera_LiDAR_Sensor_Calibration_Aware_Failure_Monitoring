"""Feature construction for calibration correction learning.

The model does not receive raw point clouds directly. Instead, we create a compact
image-aligned representation:

    RGB + sparse LiDAR depth + image edge + LiDAR-edge residual

This is fast, easy to debug visually, and suitable for TensorFlow CNN training.
"""

from __future__ import annotations

import numpy as np
import cv2

from calibguard.geometry.projection import project_lidar_to_image
from calibguard.geometry.se3 import apply_delta_to_extrinsic


def make_sparse_depth_map(proj_uvd: np.ndarray, image_shape: tuple[int, int, int], max_depth: float = 80.0) -> np.ndarray:
    """Rasterize projected LiDAR points into a sparse normalized depth map.

    Depth is stored as `1 - depth / max_depth`, so closer points are brighter.
    Pixels without LiDAR points remain 0.
    """
    h, w = image_shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    if proj_uvd.size == 0:
        return depth

    u = np.round(proj_uvd[:, 0]).astype(np.int32)
    v = np.round(proj_uvd[:, 1]).astype(np.int32)
    z = np.clip(proj_uvd[:, 2], 0.0, max_depth)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # Keep the closest point when multiple points fall into the same pixel.
    for px, py, pz in zip(u[valid], v[valid], z[valid]):
        old = depth[py, px]
        if old == 0.0 or pz < old:
            depth[py, px] = pz

    mask = depth > 0
    out = np.zeros_like(depth)
    out[mask] = 1.0 - np.clip(depth[mask] / max_depth, 0.0, 1.0)
    return out.astype(np.float32)


def make_edge_and_distance(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create image edge map and distance-to-edge map.

    The distance map is useful because if projected LiDAR points are far from image
    edges, the extrinsic calibration may be wrong.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 80, 160)
    edge01 = (edge > 0).astype(np.float32)

    # cv2.distanceTransform gives distance to the nearest zero pixel.
    # We invert the edge map so distance is computed to edge pixels.
    inv = ((1.0 - edge01) * 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    dist = np.clip(dist / 32.0, 0.0, 1.0).astype(np.float32)
    return edge01, dist


def make_residual_sparse_map(proj_uvd: np.ndarray, dist_map: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Create sparse residual map sampled from the image edge distance map.

    Each projected LiDAR pixel stores its distance-to-nearest-image-edge value.
    Larger values indicate worse local image-LiDAR alignment.
    """
    h, w = image_shape[:2]
    residual = np.zeros((h, w), dtype=np.float32)
    if proj_uvd.size == 0:
        return residual
    u = np.round(proj_uvd[:, 0]).astype(np.int32)
    v = np.round(proj_uvd[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    residual[v[valid], u[valid]] = dist_map[v[valid], u[valid]]
    return residual.astype(np.float32)


def resize_and_stack(rgb: np.ndarray, depth: np.ndarray, edge: np.ndarray, residual: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize feature maps and concatenate into a 6-channel tensor."""
    rgb_r = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    depth_r = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)[..., None]
    edge_r = cv2.resize(edge, (out_w, out_h), interpolation=cv2.INTER_NEAREST)[..., None]
    residual_r = cv2.resize(residual, (out_w, out_h), interpolation=cv2.INTER_NEAREST)[..., None]
    return np.concatenate([rgb_r, depth_r, edge_r, residual_r], axis=-1).astype(np.float32)


def build_model_input(
    rgb: np.ndarray,
    lidar: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    Tr_velo_to_cam: np.ndarray,
    drift_vec6: np.ndarray,
    out_h: int,
    out_w: int,
    max_depth: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 6-channel model input from a frame and synthetic drift.

    Returns
    -------
    x:
        6-channel model input, shape `(out_h, out_w, 6)`.
    proj_bad:
        Drifted projected LiDAR points `[u, v, depth, reflectance]`, useful for
        visualization and metric computation.
    """
    T_bad = apply_delta_to_extrinsic(Tr_velo_to_cam, drift_vec6)
    proj_bad = project_lidar_to_image(lidar, P2, R0_rect, T_bad, rgb.shape)
    depth = make_sparse_depth_map(proj_bad, rgb.shape, max_depth=max_depth)
    edge, dist = make_edge_and_distance(rgb)
    residual = make_residual_sparse_map(proj_bad, dist, rgb.shape)
    x = resize_and_stack(rgb, depth, edge, residual, out_h, out_w)
    return x, proj_bad
