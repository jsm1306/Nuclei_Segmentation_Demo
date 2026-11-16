from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )
        self.norm2 = layers.LayerNormalization()
        self.mlp = tf.keras.Sequential(
            [
                # The tanh approximation exports to portable ONNX operators
                # (exact GELU becomes Erfc, which ONNX Runtime cannot load).
                layers.Dense(
                    mlp_dim,
                    activation=lambda value: tf.keras.activations.gelu(
                        value, approximate=True
                    ),
                ),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        y = self.norm1(x)
        y = self.attn(y, y, training=training)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y, training=training)
        return x + y


def transformer_bridge(x: tf.Tensor) -> tf.Tensor:
    shortcut = x
    x = layers.Conv2D(384, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    height, width, channels = x.shape[1], x.shape[2], x.shape[-1]
    x = layers.Reshape((height * width, channels))(x)
    x = TransformerBlock(embed_dim=384, num_heads=8, mlp_dim=768)(x)
    x = TransformerBlock(embed_dim=384, num_heads=8, mlp_dim=768)(x)
    x = layers.Reshape((height, width, channels))(x)
    x = layers.Conv2D(576, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    return layers.Activation("relu")(x)


def attention_gate(
    skip: tf.Tensor, gating: tf.Tensor, inter_channels: int
) -> tf.Tensor:
    theta_x = layers.Conv2D(inter_channels, kernel_size=1, padding="same")(skip)
    phi_g = layers.Conv2D(inter_channels, kernel_size=1, padding="same")(gating)
    add = layers.Add()([theta_x, phi_g])
    add = layers.Activation("relu")(add)
    psi = layers.Conv2D(
        1, kernel_size=1, padding="same", activation="sigmoid"
    )(add)
    return layers.Multiply()([skip, psi])


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    skip = attention_gate(skip, x, inter_channels=filters // 2)
    x = layers.Concatenate()([x, skip])
    return conv_block(x, filters)


def reverse_attention_module(
    feature_map: tf.Tensor, coarse_mask: tf.Tensor
) -> tf.Tensor:
    reverse_map = layers.Lambda(lambda value: 1.0 - value)(coarse_mask)
    reverse_features = layers.Multiply()([feature_map, reverse_map])
    reverse_features = layers.Conv2D(
        feature_map.shape[-1], 3, padding="same", activation="relu"
    )(reverse_features)
    refined = layers.Add()([feature_map, reverse_features])
    return conv_block(refined, 64)


def build_nuraformer(input_shape: tuple[int, int, int] = (512, 512, 3)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=None,
    )
    skip1 = backbone.get_layer("block_1_expand_relu").output
    skip2 = backbone.get_layer("block_3_expand_relu").output
    skip3 = backbone.get_layer("block_6_expand_relu").output
    bottleneck = transformer_bridge(
        backbone.get_layer("block_13_expand_relu").output
    )

    d1 = decoder_block(bottleneck, skip3, 256)
    d2 = decoder_block(d1, skip2, 128)
    d3 = decoder_block(d2, skip1, 64)
    d4 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d3)
    d4 = conv_block(d4, 32)
    coarse_mask = layers.Conv2D(1, 1, activation="sigmoid", name="coarse_mask")(d4)
    refined_features = reverse_attention_module(d4, coarse_mask)
    refined_features = conv_block(refined_features, 64)
    seg_mask = layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32", name="seg_mask"
    )(refined_features)
    boundary_features = layers.Conv2D(
        16, 3, padding="same", activation="relu"
    )(refined_features)
    boundary_mask = layers.Conv2D(
        1, 1, activation="sigmoid", dtype="float32", name="boundary_mask"
    )(boundary_features)
    return models.Model(inputs, [seg_mask, boundary_mask])


def load_model(weights_path: str) -> tf.keras.Model:
    model = build_nuraformer()
    model.load_weights(weights_path)
    return model
