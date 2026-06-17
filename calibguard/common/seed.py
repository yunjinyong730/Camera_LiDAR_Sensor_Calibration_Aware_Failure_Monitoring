"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow seeds.

    Full determinism is not guaranteed for every GPU kernel, but this is enough
    for stable debugging and comparable toy experiments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
