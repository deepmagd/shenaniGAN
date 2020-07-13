import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from typing import Function


class ResidualLayer(layers.Layer):
    def __init__(
        self,
        filters_in: int,
        filters_out: int,
        w_init: tf.Tensor,
        bn_init: tf.Tensor,
        activation: Function,
        first_conv_pad: str="valid"
    ):
        super(ResidualLayer, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.w_init = w_init
        self.bn_init = bn_init
        self.activation = activation
        self.first_conv_pad = first_conv_pad

    def build(self, input_shape):
        self.conv2d_1 = Conv2D(
            filters=self.filters_in,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=self.first_conv_pad,
            kernel_initializer=self.w_init,
        )
        self.bn_1 = BatchNormalization(gamma_initializer=self.bn_init)

        self.conv2d_2 = Conv2D(
            filters=self.filters_in,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=self.w_init,
        )
        self.bn_2 = BatchNormalization(gamma_initializer=self.bn_init)

        self.conv2d_3 = Conv2D(
            filters=self.filters_out,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer=self.w_init,
        )
        self.bn_3 = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x: tf.Tensor, training: bool = True):
        x = self.conv2d_1(x)
        x = self.bn_1(x, training=training)
        x = self.activation(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = self.activation(x)

        x = self.conv2d_3(x)
        return self.bn_3(x, training=training)


class ConditionalAugmentation(layers.Layer):
    def __init__(self, conditional_emb_size: int, w_init: tf.Tensor):
        super(ConditionalAugmentation, self).__init__()
        self.conditional_emb_size = conditional_emb_size
        self.w_init = w_init

    def build(self, input_shape):
        self.dense_mean = Dense(
            units=self.conditional_emb_size, kernel_initializer=self.w_init
        )
        self.dense_sigma = Dense(
            units=self.conditional_emb_size, kernel_initializer=self.w_init
        )

    def call(self, embedding: tf.Tensor):
        mean = tf.nn.leaky_relu(self.dense_mean(embedding), alpha=0.2)
        log_sigma = tf.nn.leaky_relu(self.dense_sigma(embedding), alpha=0.2)
        epsilon = tf.random.truncated_normal(tf.shape(mean))
        stddev = tf.math.exp(log_sigma)
        smoothed_embedding = mean + stddev * epsilon
        return smoothed_embedding, mean, log_sigma
