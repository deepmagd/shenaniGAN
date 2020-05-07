import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense)


class ResidualLayer(layers.Layer):

    def __init__(self, filters_in, filters_out, w_init, bn_init):
        super(ResidualLayer, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, input_shape):
        self.conv2d_1 = Conv2D(filters=self.filters_in, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=self.w_init)
        self.bn_1 = BatchNormalization(gamma_initializer=self.bn_init)

        self.conv2d_2 = Conv2D(filters=self.filters_in, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.w_init)
        self.bn_2 = BatchNormalization(gamma_initializer=self.bn_init)

        self.conv2d_3 = Conv2D(filters=self.filters_out, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.w_init)
        self.bn_3 = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        x = self.conv2d_1(x)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_3(x)
        return self.bn_3(x, training=training)

class DeconvBlock(layers.Layer):

    def __init__(self, filters, w_init, bn_init, activation=False):
        super(DeconvBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, inout_shape):
        self.deconv2d = Conv2DTranspose(self.filters, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=self.w_init)
        self.conv2d = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=self.w_init)
        self.bn = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        x = self.deconv2d(x)
        x = self.conv2d(x)
        x = self.bn(x, training=training)
        if self.activation:
            x = tf.nn.relu(x)
        return x

class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding, w_init, bn_init, activation=False):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, input_shape):
        self.conv2d = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=self.w_init, use_bias=False)
        self.bn = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        x = self.conv2d(x)
        x = self.bn(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

class ConditionalAugmentation(layers.Layer):

    def __init__(self, conditional_emb_size, w_init):
        super(ConditionalAugmentation, self).__init__()
        self.conditional_emb_size = conditional_emb_size
        self.w_init = w_init

    def build(self, inout_shape):
        self.dense_mean = Dense(units=self.conditional_emb_size, kernel_initializer=self.w_init)
        self.dense_sigma = Dense(units=self.conditional_emb_size, kernel_initializer=self.w_init)

    def call(self, embedding):
        mean = tf.nn.leaky_relu(self.dense_mean(embedding), alpha=0.2)
        log_sigma = tf.nn.leaky_relu(self.dense_sigma(embedding), alpha=0.2)
        epsilon = tf.random.truncated_normal(tf.shape(mean))
        stddev = tf.math.exp(log_sigma)
        smoothed_embedding = mean + stddev * epsilon
        return smoothed_embedding, mean, log_sigma
