"""Visualization helpers for projection and recovery demos."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def colorize_depth(depth: np.ndarray, max_depth: float = 80.0) -> np.ndarray:
    """Map depth values to OpenCV colors for readable projection overlays."""
    d = np.clip(depth / max_depth, 0.0, 1.0)
    vals = (255 * (1.0 - d)).astype(np.uint8)
    colors = cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return colors


def draw_projection(rgb: np.ndarray, proj_uvd: np.ndarray, radius: int = 1, max_depth: float = 80.0) -> np.ndarray:
    """Overlay projected LiDAR points on an RGB image.

    Returns RGB image.
    """
    out = rgb.copy()
    if proj_uvd.size == 0:
        return out
    colors_bgr = colorize_depth(proj_uvd[:, 2], max_depth=max_depth)
    # Convert BGR color map to RGB before drawing on RGB image.
    colors_rgb = colors_bgr[:, ::-1]
    h, w = out.shape[:2]
    for (u, v, _z, _r), color in zip(proj_uvd, colors_rgb):
        x, y = int(round(u)), int(round(v))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, tuple(int(c) for c in color), -1)
    return out


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """Save an RGB image using OpenCV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)


def make_horizontal_comparison(images: list[np.ndarray], titles: list[str] | None = None) -> np.ndarray:
    """Concatenate images horizontally with optional title text."""
    if not images:
        raise ValueError("images must not be empty")
    h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] != h:
            scale = h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), h))
        resized.append(img)
    canvas = np.concatenate(resized, axis=1)
    if titles:
        x_offset = 10
        for img, title in zip(resized, titles):
            cv2.putText(canvas, title, (x_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset += img.shape[1]
    return canvas


def put_text_box(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.65, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """Draw readable text with a dark outline on an RGB image."""
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_tricolor_projection(
    rgb: np.ndarray,
    uv_gt: np.ndarray,
    uv_bad: np.ndarray,
    uv_rec: np.ndarray,
    radius: int = 2,
    max_points: int = 1200,
) -> np.ndarray:
    """Overlay same LiDAR points under three transforms.

    Color convention:
      - Green: normal/ground-truth calibration
      - Red: drifted calibration
      - Cyan: recovered calibration

    If recovery works, cyan points should move closer to green points than red
    points. This directly solves the "I cannot tell if it recovered" problem.
    """
    out = rgb.copy()
    if len(uv_gt) == 0:
        return out
    n = min(len(uv_gt), max_points)
    idx = np.linspace(0, len(uv_gt) - 1, n).astype(np.int64)
    for arr, color in [(uv_bad, (255, 45, 45)), (uv_rec, (0, 230, 255)), (uv_gt, (0, 255, 0))]:
        for u, v in arr[idx, :2]:
            cv2.circle(out, (int(round(u)), int(round(v))), radius, color, -1, lineType=cv2.LINE_AA)
    put_text_box(out, "Green=Normal  Red=Drifted  Cyan=Recovered", (12, 28), scale=0.65)
    return out


def draw_error_vectors(
    rgb: np.ndarray,
    uv_gt: np.ndarray,
    uv_bad: np.ndarray,
    uv_rec: np.ndarray,
    max_vectors: int = 250,
) -> np.ndarray:
    """Draw error vectors from normal projection to drifted/recovered projection.

    Red line length represents drift error. Cyan line length represents recovered
    error. Recovery is visually clear when cyan lines are much shorter than red.
    """
    out = rgb.copy()
    if len(uv_gt) == 0:
        return out
    n = min(len(uv_gt), max_vectors)
    idx = np.linspace(0, len(uv_gt) - 1, n).astype(np.int64)
    for i in idx:
        g = tuple(np.round(uv_gt[i, :2]).astype(int))
        b = tuple(np.round(uv_bad[i, :2]).astype(int))
        r = tuple(np.round(uv_rec[i, :2]).astype(int))
        cv2.line(out, g, b, (255, 40, 40), 1, cv2.LINE_AA)
        cv2.line(out, g, r, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.circle(out, g, 2, (0, 255, 0), -1, cv2.LINE_AA)
    put_text_box(out, "Error vectors: Normal->Drifted(red), Normal->Recovered(cyan)", (12, 28), scale=0.62)
    return out


def make_metric_panel(width: int, height: int, metrics: dict, title: str = "CalibGuard Recovery Report") -> np.ndarray:
    """Create a dashboard-like metric panel as an RGB image."""
    panel = np.full((height, width, 3), 25, dtype=np.uint8)
    put_text_box(panel, title, (20, 38), scale=0.78, color=(255, 255, 255))

    drift_err = float(metrics.get("drifted_reprojection_error_px", float("nan")))
    rec_err = float(metrics.get("recovered_reprojection_error_px", float("nan")))
    rate = float(metrics.get("recovery_rate_percent", float("nan")))
    status = str(metrics.get("recovery_status", "UNKNOWN"))
    conf = metrics.get("predicted_confidence", None)
    mode = str(metrics.get("recovery_mode", "model"))

    status_color = (0, 255, 0) if status == "RECOVERED" else (255, 220, 0) if status == "PARTIALLY_RECOVERED" else (255, 70, 70)
    put_text_box(panel, f"Status: {status}", (20, 82), scale=0.72, color=status_color)
    put_text_box(panel, f"Mode: {mode}", (20, 116), scale=0.58, color=(220, 220, 220))

    rows = [
        ("Drift error", drift_err, (255, 70, 70)),
        ("Recovered error", rec_err, (0, 230, 255)),
    ]
    max_err = max([v for _, v, _ in rows if np.isfinite(v)] + [1.0])
    x0, y0, bar_w, bar_h = 20, 155, width - 60, 24
    for j, (name, val, color) in enumerate(rows):
        y = y0 + j * 58
        put_text_box(panel, f"{name}: {val:.2f} px", (x0, y - 8), scale=0.55, color=(240, 240, 240))
        cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + bar_h), (70, 70, 70), 1)
        fill = int(bar_w * min(1.0, max(0.0, val / max_err))) if np.isfinite(val) else 0
        cv2.rectangle(panel, (x0, y), (x0 + fill, y + bar_h), color, -1)

    put_text_box(panel, f"Recovery rate: {rate:.1f}%", (20, y0 + 130), scale=0.7, color=status_color)
    if conf is not None:
        put_text_box(panel, f"Model confidence: {float(conf):.3f}", (20, y0 + 168), scale=0.58, color=(220, 220, 220))

    drift = metrics.get("injected_drift", None)
    corr = metrics.get("predicted_correction", None)
    final_corr = metrics.get("final_correction", corr)
    if drift is not None:
        put_text_box(panel, "Injected drift [r,p,y,tx,ty,tz]", (20, height - 108), scale=0.48, color=(200, 200, 200))
        put_text_box(panel, np.array2string(np.asarray(drift), precision=3, suppress_small=True), (20, height - 78), scale=0.48, color=(255, 180, 180))
    if final_corr is not None:
        put_text_box(panel, "Final correction [r,p,y,tx,ty,tz]", (20, height - 44), scale=0.48, color=(200, 200, 200))
        put_text_box(panel, np.array2string(np.asarray(final_corr), precision=3, suppress_small=True), (20, height - 14), scale=0.48, color=(180, 255, 255))
    return panel


def resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize RGB image to `(width, height)`."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_recovery_report_image(
    normal_img: np.ndarray,
    drifted_img: np.ndarray,
    recovered_img: np.ndarray,
    tricolor_img: np.ndarray,
    vectors_img: np.ndarray,
    metrics: dict,
    panel_width: int = 520,
) -> np.ndarray:
    """Build a single image that clearly explains whether recovery worked.

    Layout:
        row 1: Normal | Drifted | Recovered
        row 2: Tri-color overlay | Error vectors | Metric panel
    """
    h, w = normal_img.shape[:2]
    top = make_horizontal_comparison(
        [normal_img, drifted_img, recovered_img],
        ["Normal projection", "Drifted projection", "Recovered projection"],
    )
    tri = resize_to(tricolor_img, (w, h))
    vec = resize_to(vectors_img, (w, h))
    panel = make_metric_panel(panel_width, h, metrics)
    bottom = np.concatenate([tri, vec, panel], axis=1)
    if bottom.shape[1] != top.shape[1]:
        # Match widths by padding the shorter row. This keeps video writing simple.
        target_w = max(bottom.shape[1], top.shape[1])
        def pad(img):
            if img.shape[1] == target_w:
                return img
            pad_w = target_w - img.shape[1]
            return np.pad(img, ((0, 0), (0, pad_w), (0, 0)), mode="constant", constant_values=25)
        top = pad(top)
        bottom = pad(bottom)
    return np.concatenate([top, bottom], axis=0)


def write_mp4(path: str | Path, frames: list[np.ndarray], fps: int = 4) -> None:
    """Write RGB frames to an MP4 video with OpenCV."""
    if not frames:
        raise ValueError("frames must not be empty")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {p}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

