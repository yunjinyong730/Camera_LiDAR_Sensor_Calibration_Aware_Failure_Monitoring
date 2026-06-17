"""TensorFlow implementation of MSF-CalibNet.

The model is intentionally lightweight and fully Keras/TensorFlow compatible.
It is designed for camera-LiDAR calibration correction rather than generic image
classification.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
def hard_swish(x):
    """Mobile-friendly activation used in many efficient CNNs."""
    return x * tf.nn.relu6(x + 3.0) / 6.0


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
class ConvBNAct(layers.Layer):
    """Conv2D + BatchNorm + optional hard-swish activation."""

    def __init__(self, filters: int, kernel: int = 3, stride: int = 1, groups: int = 1, act: bool = True, **kwargs):
        # Keras passes generic layer arguments such as name, trainable, and dtype
        # during model loading. Accepting **kwargs makes this custom layer
        # safely deserializable from .keras files.
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel = int(kernel)
        self.stride = int(stride)
        self.groups = int(groups)
        self.use_act = bool(act)
        self.conv = layers.Conv2D(
            filters,
            kernel,
            strides=stride,
            padding="same",
            use_bias=False,
            groups=groups,
            kernel_initializer="he_normal",
        )
        self.bn = layers.BatchNormalization()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters, "kernel": self.kernel, "stride": self.stride, "groups": self.groups, "act": self.use_act})
        return cfg

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.use_act:
            x = hard_swish(x)
        return x


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
class SqueezeExciteLite(layers.Layer):
    """Small squeeze-and-excitation block.

    It adds channel attention with low parameter overhead. This helps the model
    decide whether RGB, depth, edge, or residual channels are more informative.
    """

    def __init__(self, channels: int, se_ratio: float = 0.125, **kwargs):
        # Keras deserialization may provide name/trainable/dtype here.
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.se_ratio = float(se_ratio)
        hidden = max(8, int(channels * se_ratio))
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(hidden, 1, activation="relu")
        self.fc2 = layers.Conv2D(channels, 1, activation="sigmoid")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "se_ratio": self.se_ratio})
        return cfg

    def call(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.fc2(s)
        return x * s


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
class FusedIRBlock(layers.Layer):
    """Fused inverted residual block.

    Early feature maps are large. On many devices, a fused 3x3 conv can be faster
    than splitting into pointwise + depthwise operations too early.
    """

    def __init__(self, out_ch: int, expansion: int = 2, stride: int = 1, se: bool = False, **kwargs):
        # Accept name/trainable/dtype from serialized Keras configs.
        super().__init__(**kwargs)
        self.out_ch = int(out_ch)
        self.expansion = int(expansion)
        self.stride = int(stride)
        self.se_flag = bool(se)
        self.exp_ch = out_ch * expansion
        self.conv1 = ConvBNAct(self.exp_ch, kernel=3, stride=stride)
        self.se = SqueezeExciteLite(self.exp_ch) if se else None
        self.conv2 = ConvBNAct(out_ch, kernel=1, stride=1, act=False)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_ch": self.out_ch, "expansion": self.expansion, "stride": self.stride, "se": self.se_flag})
        return cfg

    def build(self, input_shape):
        self.use_residual = self.stride == 1 and int(input_shape[-1]) == self.out_ch

    def call(self, x, training=False):
        y = self.conv1(x, training=training)
        if self.se is not None:
            y = self.se(y)
        y = self.conv2(y, training=training)
        return x + y if self.use_residual else y


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
class DWIRBlock(layers.Layer):
    """Depthwise inverted residual block.

    This is the main efficient CNN block. It reduces computation while preserving
    spatial alignment cues needed for calibration.
    """

    def __init__(self, out_ch: int, expansion: int = 4, stride: int = 1, kernel: int = 3, se: bool = False, **kwargs):
        # Accept name/trainable/dtype from serialized Keras configs.
        super().__init__(**kwargs)
        self.out_ch = int(out_ch)
        self.expansion = int(expansion)
        self.stride = int(stride)
        self.kernel = int(kernel)
        self.use_se = bool(se)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_ch": self.out_ch, "expansion": self.expansion, "stride": self.stride, "kernel": self.kernel, "se": self.use_se})
        return cfg

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        exp_ch = in_ch * self.expansion
        self.use_residual = self.stride == 1 and in_ch == self.out_ch
        self.pw1 = ConvBNAct(exp_ch, kernel=1, stride=1)
        self.dw = layers.DepthwiseConv2D(self.kernel, strides=self.stride, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.se = SqueezeExciteLite(exp_ch) if self.use_se else None
        self.pw2 = ConvBNAct(self.out_ch, kernel=1, stride=1, act=False)

    def call(self, x, training=False):
        y = self.pw1(x, training=training)
        y = self.dw(y)
        y = self.bn(y, training=training)
        y = hard_swish(y)
        if self.se is not None:
            y = self.se(y)
        y = self.pw2(y, training=training)
        return x + y if self.use_residual else y


@tf.keras.utils.register_keras_serializable(package="CalibGuard")
class StripDWBlock(layers.Layer):
    """Large receptive field block using factorized depthwise convolutions.

    Projection mismatch often appears as structured horizontal/vertical shifts.
    `1 x K` and `K x 1` depthwise convolutions enlarge the receptive field while
    keeping computation low.
    """

    def __init__(self, out_ch: int, kernel: int = 7, stride: int = 1, **kwargs):
        # Accept name/trainable/dtype from serialized Keras configs.
        super().__init__(**kwargs)
        self.out_ch = int(out_ch)
        self.kernel = int(kernel)
        self.stride = int(stride)
        self.dw_h = layers.DepthwiseConv2D((1, kernel), strides=(1, stride), padding="same", use_bias=False)
        self.bn_h = layers.BatchNormalization()
        self.dw_v = layers.DepthwiseConv2D((kernel, 1), strides=(stride, 1), padding="same", use_bias=False)
        self.bn_v = layers.BatchNormalization()
        self.pw = ConvBNAct(out_ch, kernel=1, stride=1, act=False)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_ch": self.out_ch, "kernel": self.kernel, "stride": self.stride})
        return cfg

    def build(self, input_shape):
        self.use_residual = self.stride == 1 and int(input_shape[-1]) == self.out_ch

    def call(self, x, training=False):
        y = self.dw_h(x)
        y = self.bn_h(y, training=training)
        y = hard_swish(y)
        y = self.dw_v(y)
        y = self.bn_v(y, training=training)
        y = hard_swish(y)
        y = self.pw(y, training=training)
        return x + y if self.use_residual else y


def stage_pool_token(x, dim: int, name: str):
    """Convert a feature map into a compact stage token."""
    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    x = layers.Dense(dim, use_bias=False, name=f"{name}_proj")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation(hard_swish, name=f"{name}_act")(x)
    return x


def build_msf_calibnet(
    input_shape: tuple[int, int, int] = (192, 640, 6),
    width_mult: float = 1.0,
    token_dim: int = 96,
    dropout: float = 0.1,
) -> tf.keras.Model:
    """Build MSF-CalibNet.

    Parameters
    ----------
    input_shape:
        `(height, width, 6)` model input shape.
    width_mult:
        Channel multiplier. Use 0.5 or 0.75 for faster experiments.
    token_dim:
        Dimension of each multi-stage pooled token.
    dropout:
        Dropout in the final regression head.
    """

    def c(ch: int) -> int:
        return max(8, int(ch * width_mult))

    inp = layers.Input(shape=input_shape, name="calib_input")

    # Stem: reduce spatial resolution while preserving low-level edge/depth cues.
    x = ConvBNAct(c(24), kernel=3, stride=2, name="stem")(inp)
    x = FusedIRBlock(c(24), expansion=2, stride=1, name="fused_s1_b1")(x)

    # Stage 2: high-resolution local alignment features.
    x = FusedIRBlock(c(40), expansion=2, stride=2, name="fused_s2_b1")(x)
    x = FusedIRBlock(c(40), expansion=2, stride=1, name="fused_s2_b2")(x)
    s2 = x

    # Stage 3: medium-level boundary and depth structure.
    x = DWIRBlock(c(64), expansion=3, stride=2, kernel=3, name="dwir_s3_b1")(x)
    x = DWIRBlock(c(64), expansion=3, stride=1, kernel=3, name="dwir_s3_b2")(x)
    x = StripDWBlock(c(64), kernel=7, stride=1, name="strip_s3")(x)
    s3 = x

    # Stage 4: larger receptive field with light SE.
    x = DWIRBlock(c(96), expansion=4, stride=2, kernel=5, se=True, name="dwir_s4_b1")(x)
    x = DWIRBlock(c(96), expansion=4, stride=1, kernel=5, se=True, name="dwir_s4_b2")(x)
    x = StripDWBlock(c(96), kernel=9, stride=1, name="strip_s4")(x)
    s4 = x

    # Stage 5: compact global alignment token.
    x = DWIRBlock(c(128), expansion=4, stride=2, kernel=5, se=True, name="dwir_s5_b1")(x)
    x = ConvBNAct(c(160), kernel=1, stride=1, name="head_conv")(x)
    s5 = x

    # Multi-stage fusion without a heavy FPN.
    # This is efficient and makes the model robust to both local and global drift.
    t2 = stage_pool_token(s2, token_dim, "token_s2")
    t3 = stage_pool_token(s3, token_dim, "token_s3")
    t4 = stage_pool_token(s4, token_dim, "token_s4")
    t5 = stage_pool_token(s5, token_dim, "token_s5")

    z = layers.Concatenate(name="multi_stage_token_fusion")([t2, t3, t4, t5])
    z = layers.Dense(192, use_bias=False, name="fusion_fc1")(z)
    z = layers.BatchNormalization(name="fusion_bn1")(z)
    z = layers.Activation(hard_swish, name="fusion_act1")(z)
    z = layers.Dropout(dropout, name="fusion_dropout")(z)
    z = layers.Dense(96, activation=hard_swish, name="fusion_fc2")(z)

    # tanh keeps normalized correction bounded in roughly [-1, 1].
    correction = layers.Dense(6, activation="tanh", name="correction")(z)
    confidence = layers.Dense(1, activation="sigmoid", name="confidence")(z)

    return tf.keras.Model(inp, {"correction": correction, "confidence": confidence}, name="MSF-CalibNet")


if __name__ == "__main__":
    model = build_msf_calibnet()
    model.summary()
