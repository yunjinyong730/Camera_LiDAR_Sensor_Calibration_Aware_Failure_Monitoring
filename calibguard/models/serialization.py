"""Model serialization helpers for Keras custom layers."""

from __future__ import annotations

import tensorflow as tf

from calibguard.models.msf_calibnet import (
    hard_swish,
    ConvBNAct,
    SqueezeExciteLite,
    FusedIRBlock,
    DWIRBlock,
    StripDWBlock,
)

CUSTOM_OBJECTS = {
    "hard_swish": hard_swish,
    "ConvBNAct": ConvBNAct,
    "SqueezeExciteLite": SqueezeExciteLite,
    "FusedIRBlock": FusedIRBlock,
    "DWIRBlock": DWIRBlock,
    "StripDWBlock": StripDWBlock,
}


def load_calibguard_model(path: str, compile: bool = False) -> tf.keras.Model:
    """Load a saved CalibGuard Keras model with all custom layers registered."""
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=compile)
