"""Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM).

Notation follows Section 3.5 of the manuscript. Both modules act on the
terminal feature map F in R^{H x W x C} of a backbone, before global pooling.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class SEBlock(layers.Layer):
    """s = sigmoid(W2 ReLU(W1 z)),  z_c = GAP(F_c),  F~_c = s_c F_c   (Eqs. (3)-(5))."""

    def __init__(self, ratio: int = 16, **kw):
        super().__init__(**kw)
        self.ratio = ratio

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.fc1 = layers.Dense(max(c // self.ratio, 1), activation="relu", name="fc1")
        self.fc2 = layers.Dense(c, activation="sigmoid", name="fc2")
        super().build(input_shape)

    def call(self, x):
        z = tf.reduce_mean(x, axis=[1, 2])                 # squeeze
        s = self.fc2(self.fc1(z))                          # excitation
        self.last_weights = s
        return x * s[:, None, None, :]

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio}


class CBAM(layers.Layer):
    """Channel attention M_c followed by spatial attention M_s (Eqs. (6)-(7)).

    M_c(F)  = sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))      shared MLP, ratio r
    M_s(F') = sigmoid(conv_{k x k}([AvgPool_c(F'); MaxPool_c(F')]))
    """

    def __init__(self, ratio: int = 16, kernel: int = 7, **kw):
        super().__init__(**kw)
        self.ratio, self.kernel = ratio, kernel

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.mlp1 = layers.Dense(max(c // self.ratio, 1), activation="relu", name="mlp1")
        self.mlp2 = layers.Dense(c, name="mlp2")
        self.conv = layers.Conv2D(1, self.kernel, padding="same", activation="sigmoid",
                                  name="spatial_conv")
        super().build(input_shape)

    def call(self, x):
        avg = tf.reduce_mean(x, axis=[1, 2])
        mx = tf.reduce_max(x, axis=[1, 2])
        mc = tf.sigmoid(self.mlp2(self.mlp1(avg)) + self.mlp2(self.mlp1(mx)))
        x1 = x * mc[:, None, None, :]
        pooled = tf.concat([tf.reduce_mean(x1, axis=-1, keepdims=True),
                            tf.reduce_max(x1, axis=-1, keepdims=True)], axis=-1)
        ms = self.conv(pooled)
        self.last_spatial = ms                              # H x W x 1 map (Figure 11)
        return x1 * ms

    def get_config(self):
        return {**super().get_config(), "ratio": self.ratio, "kernel": self.kernel}


CUSTOM_OBJECTS = {"SEBlock": SEBlock, "CBAM": CBAM}
