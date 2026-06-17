"""Lightweight geometry refinement for camera-LiDAR calibration correction.

The learned model predicts a 6DoF correction from image-aligned features. For a
clearer demo, this module optionally performs a small local search around the
model prediction using an image-edge alignment objective. This does not use the
KITTI ground-truth calibration during optimization, so it is closer to a practical
self-supervised refinement step than an oracle correction.
"""

from __future__ import annotations

import numpy as np

from calibguard.geometry.se3 import apply_delta_to_extrinsic
from calibguard.geometry.projection import project_lidar_to_image


def edge_alignment_score(
    lidar: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    T_candidate: np.ndarray,
    distance_map: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    max_points: int = 4096,
) -> float:
    """Score a candidate transform using distance-to-image-edge values.

    Smaller is better. Projected LiDAR points that fall closer to image edges
    receive a lower score. We also add a small penalty if too few points are
    visible, preventing degenerate candidates from winning because they project
    almost nothing into the image.
    """
    h, w = image_shape[:2]
    proj = project_lidar_to_image(lidar, P2, R0_rect, T_candidate, image_shape)
    if proj.size == 0:
        return 1e6
    if len(proj) > max_points:
        # Deterministic spread over the point set for reproducible evaluation.
        idx = np.linspace(0, len(proj) - 1, max_points).astype(np.int64)
        proj = proj[idx]
    u = np.round(proj[:, 0]).astype(np.int32)
    v = np.round(proj[:, 1]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(valid):
        return 1e6
    edge_cost = float(np.mean(distance_map[v[valid], u[valid]]))
    visible_penalty = 0.05 * float(1.0 / max(1, np.sum(valid)))
    return edge_cost + visible_penalty


def _candidate_offsets(stage: int) -> list[np.ndarray]:
    """Generate small local-search offsets.

    The search is intentionally small because this is a demo refinement step, not
    a full calibration solver. It focuses on yaw/pitch/tx because these produce
    visible camera-LiDAR overlay shifts in KITTI-like images.
    """
    if stage == 0:
        yaw_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        pitch_vals = [-0.6, 0.0, 0.6]
        tx_vals = [-0.08, 0.0, 0.08]
    else:
        yaw_vals = [-0.3, 0.0, 0.3]
        pitch_vals = [-0.2, 0.0, 0.2]
        tx_vals = [-0.03, 0.0, 0.03]
    offsets = []
    for pitch in pitch_vals:
        for yaw in yaw_vals:
            for tx in tx_vals:
                offsets.append(np.array([0.0, pitch, yaw, tx, 0.0, 0.0], dtype=np.float32))
    return offsets


def refine_correction_by_edge_search(
    initial_corr: np.ndarray,
    T_bad: np.ndarray,
    lidar: np.ndarray,
    P2: np.ndarray,
    R0_rect: np.ndarray,
    distance_map: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    stages: int = 2,
) -> tuple[np.ndarray, float]:
    """Refine a predicted correction with a coarse-to-fine edge search.

    Parameters
    ----------
    initial_corr:
        Model-predicted correction vector `[roll, pitch, yaw, tx, ty, tz]` in
        degrees/meters.
    T_bad:
        Drifted LiDAR-to-camera transform.

    Returns
    -------
    best_corr:
        Refined correction vector in the same units.
    best_score:
        Final edge-alignment objective value.
    """
    best_corr = np.asarray(initial_corr, dtype=np.float32).copy()
    best_T = apply_delta_to_extrinsic(T_bad, best_corr)
    best_score = edge_alignment_score(lidar, P2, R0_rect, best_T, distance_map, image_shape)

    for stage in range(stages):
        improved = False
        for offset in _candidate_offsets(stage):
            cand = best_corr + offset
            T_cand = apply_delta_to_extrinsic(T_bad, cand)
            score = edge_alignment_score(lidar, P2, R0_rect, T_cand, distance_map, image_shape)
            if score < best_score:
                best_score = score
                best_corr = cand.astype(np.float32)
                improved = True
        if not improved and stage > 0:
            break
    return best_corr.astype(np.float32), float(best_score)
