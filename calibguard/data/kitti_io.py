"""KITTI file I/O helpers.

Only the Object Detection style folder structure is required:

    training/image_2/*.png
    training/velodyne/*.bin
    training/calib/*.txt
"""

from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import cv2


def list_kitti_frames(data_root: str) -> list[str]:
    """Return sorted frame ids available in `image_2`."""
    image_dir = Path(data_root) / "image_2"
    frames = sorted(Path(p).stem for p in glob.glob(str(image_dir / "*.png")))
    if not frames:
        raise FileNotFoundError(f"No .png frames found in {image_dir}")
    return frames


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_velodyne_bin(path: str | Path) -> np.ndarray:
    """Load KITTI Velodyne `.bin` point cloud.

    KITTI stores each point as float32 `[x, y, z, reflectance]`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Velodyne file does not exist: {p}")
    arr = np.fromfile(str(p), dtype=np.float32)
    if arr.size % 4 != 0:
        raise ValueError(f"Invalid KITTI velodyne file size: {p}")
    return arr.reshape(-1, 4)


def read_kitti_calib(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read KITTI calibration matrices P2, R0_rect, Tr_velo_to_cam.

    Returns
    -------
    P2:
        3x4 projection matrix for image_2.
    R0_rect:
        3x3 rectification matrix.
    Tr_velo_to_cam:
        3x4 LiDAR-to-camera extrinsic.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {p}")

    data: dict[str, np.ndarray] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            nums = np.array([float(x) for x in value.strip().split()], dtype=np.float32)
            data[key] = nums

    if "P2" not in data:
        raise KeyError(f"P2 not found in {p}")
    if "Tr_velo_to_cam" not in data and "Tr_velo_cam" not in data:
        raise KeyError(f"Tr_velo_to_cam not found in {p}")

    P2 = data["P2"].reshape(3, 4).astype(np.float32)
    R0 = data.get("R0_rect", data.get("R_rect", np.eye(3, dtype=np.float32).reshape(-1))).reshape(3, 3).astype(np.float32)
    Tr = data.get("Tr_velo_to_cam", data.get("Tr_velo_cam")).reshape(3, 4).astype(np.float32)
    return P2, R0, Tr


def frame_paths(data_root: str, frame_id: str) -> dict[str, Path]:
    """Return standard KITTI file paths for a frame id."""
    root = Path(data_root)
    return {
        "image": root / "image_2" / f"{frame_id}.png",
        "lidar": root / "velodyne" / f"{frame_id}.bin",
        "calib": root / "calib" / f"{frame_id}.txt",
    }
