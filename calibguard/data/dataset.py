"""TensorFlow/Keras dataset for synthetic calibration drift learning."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from calibguard.data.kitti_io import list_kitti_frames, frame_paths, read_image_rgb, load_velodyne_bin, read_kitti_calib
from calibguard.features.input_builder import build_model_input
from calibguard.geometry.se3 import normalize_vec6, inverse_vec6


DEFAULT_DRIFT_SCALE = np.array([2.0, 2.0, 3.0, 0.20, 0.10, 0.20], dtype=np.float32)


def train_val_split(frames: list[str], val_ratio: float = 0.15, seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministically split frame ids into train and validation sets."""
    rng = np.random.default_rng(seed)
    arr = np.array(frames)
    rng.shuffle(arr)
    n_val = max(1, int(len(arr) * val_ratio))
    return arr[n_val:].tolist(), arr[:n_val].tolist()


class KittiCalibrationSequence(tf.keras.utils.Sequence):
    """Keras Sequence that synthesizes calibration drift on the fly.

    This design avoids saving hundreds of thousands of drifted samples to disk.
    Every time a frame is sampled, a new random drift is generated and the target
    correction is automatically known.

    Target convention:
        T_bad = Delta(drift) @ T_gt
        target_correction = inverse_SE3(drift)

    The exact inverse is used instead of simply negating the drift. This matters
    when rotation and translation are mixed.
    """

    def __init__(
        self,
        data_root: str,
        frame_ids: list[str],
        batch_size: int = 8,
        steps_per_epoch: int = 1000,
        img_h: int = 192,
        img_w: int = 640,
        drift_scale: np.ndarray = DEFAULT_DRIFT_SCALE,
        max_depth: float = 80.0,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.frame_ids = list(frame_ids)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.drift_scale = drift_scale.astype(np.float32)
        self.max_depth = float(max_depth)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def sample_drift(self) -> np.ndarray:
        """Sample a random 6DoF perturbation within configured limits."""
        return self.rng.uniform(-self.drift_scale, self.drift_scale).astype(np.float32)

    def make_sample_with_drift(self, frame_id: str, drift: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one training/evaluation sample using a provided drift vector.

        This method is also used by evaluation so that the image/LiDAR frame and
        the sampled drift stay synchronized.
        """
        paths = frame_paths(self.data_root, frame_id)
        rgb = read_image_rgb(paths["image"])
        lidar = load_velodyne_bin(paths["lidar"])
        P2, R0, Tr = read_kitti_calib(paths["calib"])

        x, _ = build_model_input(
            rgb=rgb,
            lidar=lidar,
            P2=P2,
            R0_rect=R0,
            Tr_velo_to_cam=Tr,
            drift_vec6=drift,
            out_h=self.img_h,
            out_w=self.img_w,
            max_depth=self.max_depth,
        )

        # Exact inverse correction label in SE(3), normalized for stable training.
        correction_vec6 = inverse_vec6(drift)
        correction = normalize_vec6(correction_vec6, self.drift_scale)

        # Synthetic confidence: stronger drift means lower confidence.
        # This is a supervised monitoring target, not a ground-truth sensor-health
        # label from KITTI. Later versions can replace it with metric-derived labels.
        severity = float(np.mean(np.abs(drift / self.drift_scale)))
        confidence = np.array([np.exp(-2.5 * severity)], dtype=np.float32)
        return x, correction.astype(np.float32), confidence.astype(np.float32)

    def make_sample(self, frame_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build one training sample with a randomly sampled synthetic drift."""
        drift = self.sample_drift()
        return self.make_sample_with_drift(frame_id, drift)

    def __getitem__(self, idx: int):
        xs, ys_corr, ys_conf = [], [], []
        for _ in range(self.batch_size):
            frame_id = self.rng.choice(self.frame_ids)
            x, corr, conf = self.make_sample(str(frame_id))
            xs.append(x)
            ys_corr.append(corr)
            ys_conf.append(conf)
        return np.stack(xs, axis=0), {
            "correction": np.stack(ys_corr, axis=0),
            "confidence": np.stack(ys_conf, axis=0),
        }


def make_train_val_sequences(
    data_root: str,
    batch_size: int,
    steps_per_epoch: int,
    val_steps: int,
    img_h: int,
    img_w: int,
    val_ratio: float = 0.15,
    drift_scale: np.ndarray = DEFAULT_DRIFT_SCALE,
    max_depth: float = 80.0,
    seed: int = 42,
) -> tuple[KittiCalibrationSequence, KittiCalibrationSequence, list[str], list[str]]:
    """Create train/validation Keras Sequences from a KITTI root."""
    frames = list_kitti_frames(data_root)
    train_frames, val_frames = train_val_split(frames, val_ratio=val_ratio, seed=seed)
    train_seq = KittiCalibrationSequence(
        data_root, train_frames, batch_size, steps_per_epoch, img_h, img_w,
        drift_scale=drift_scale, max_depth=max_depth, seed=seed,
    )
    val_seq = KittiCalibrationSequence(
        data_root, val_frames, batch_size, val_steps, img_h, img_w,
        drift_scale=drift_scale, max_depth=max_depth, seed=seed + 999,
    )
    return train_seq, val_seq, train_frames, val_frames
