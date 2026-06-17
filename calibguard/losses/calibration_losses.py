"""Loss functions for calibration correction learning."""

from __future__ import annotations

import tensorflow as tf


class WeightedCorrectionHuber(tf.keras.losses.Loss):
    """Huber loss for normalized 6DoF correction.

    Rotation and translation have different physical units. Targets are normalized
    before training, but giving translation a slightly higher weight often helps
    because translation drift is numerically smaller and visually important.
    """

    def __init__(self, rot_weight: float = 1.0, trans_weight: float = 1.5, delta: float = 0.05, name: str = "weighted_correction_huber"):
        super().__init__(name=name)
        self.weights = tf.constant([rot_weight, rot_weight, rot_weight, trans_weight, trans_weight, trans_weight], dtype=tf.float32)
        self.delta = float(delta)

    def call(self, y_true, y_pred):
        err = (y_true - y_pred) * self.weights
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, self.delta)
        linear = abs_err - quadratic
        loss = 0.5 * tf.square(quadratic) + self.delta * linear
        return tf.reduce_mean(loss, axis=-1)


def correction_mae_norm(y_true, y_pred):
    """Mean absolute error in normalized 6DoF target space."""
    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
