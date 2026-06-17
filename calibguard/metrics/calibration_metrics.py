"""Evaluation metrics for calibration correction."""

from __future__ import annotations

import time
import numpy as np
import tensorflow as tf

from calibguard.geometry.projection import project_lidar_to_image, mean_corresponding_reprojection_error
from calibguard.geometry.se3 import apply_delta_to_extrinsic


def rotation_error_deg(pred_corr: np.ndarray, target_corr: np.ndarray) -> float:
    """Mean absolute rotation error over roll/pitch/yaw in degrees."""
    return float(np.mean(np.abs(pred_corr[:3] - target_corr[:3])))


def translation_error_m(pred_corr: np.ndarray, target_corr: np.ndarray) -> float:
    """Mean absolute translation error over tx/ty/tz in meters."""
    return float(np.mean(np.abs(pred_corr[3:] - target_corr[3:])))


def recovered_reprojection_error(
    lidar: np.ndarray,
    image_shape: tuple[int, int, int],
    P2: np.ndarray,
    R0: np.ndarray,
    Tr_gt: np.ndarray,
    drift_vec6: np.ndarray,
    pred_corr_vec6: np.ndarray,
) -> dict[str, float]:
    """Compare normal, drifted, and recovered projections.

    Returns a dictionary with reprojection errors in pixels.
    """
    T_gt = apply_delta_to_extrinsic(Tr_gt, np.zeros(6, dtype=np.float32))
    T_bad = apply_delta_to_extrinsic(Tr_gt, drift_vec6)
    T_rec = apply_delta_to_extrinsic(T_bad, pred_corr_vec6)

    proj_gt = project_lidar_to_image(lidar, P2, R0, T_gt, image_shape)
    proj_bad = project_lidar_to_image(lidar, P2, R0, T_bad, image_shape)
    proj_rec = project_lidar_to_image(lidar, P2, R0, T_rec, image_shape)

    return {
        "drifted_reproj_error_px": mean_corresponding_reprojection_error(lidar, P2, R0, T_gt, T_bad, image_shape),
        "recovered_reproj_error_px": mean_corresponding_reprojection_error(lidar, P2, R0, T_gt, T_rec, image_shape),
        "num_gt_points": float(len(proj_gt)),
        "num_bad_points": float(len(proj_bad)),
        "num_rec_points": float(len(proj_rec)),
    }


def measure_latency_ms(model: tf.keras.Model, sample: np.ndarray, warmup: int = 5, repeat: int = 20) -> float:
    """Measure average model inference latency in milliseconds."""
    x = sample.astype(np.float32)
    for _ in range(warmup):
        _ = model(x, training=False)
    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(x, training=False)
    end = time.perf_counter()
    return float((end - start) * 1000.0 / repeat)
