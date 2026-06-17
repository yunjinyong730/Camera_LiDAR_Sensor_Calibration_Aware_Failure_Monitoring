"""TFLite export helpers."""

from __future__ import annotations

from typing import Callable, Iterable
import tensorflow as tf


def export_dynamic_range_tflite(model: tf.keras.Model, out_path: str) -> None:
    """Export a dynamic-range quantized TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def export_int8_tflite(model: tf.keras.Model, representative_dataset: Callable, out_path: str, keep_float_io: bool = True) -> None:
    """Export an int8 quantized TFLite model.

    `keep_float_io=True` makes integration easier because model input/output remain
    float32 while internal operations are quantized when possible.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if keep_float_io:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
