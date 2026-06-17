"""Small profiling utilities."""

from __future__ import annotations

import os
import tensorflow as tf


def count_params(model: tf.keras.Model) -> int:
    """Return number of trainable + non-trainable parameters."""
    return int(model.count_params())


def file_size_kb(path: str) -> float:
    """Return file size in KB."""
    return os.path.getsize(path) / 1024.0
