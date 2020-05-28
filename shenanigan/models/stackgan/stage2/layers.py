import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D

class ResidualLayerStage2(layers.Layer):

    def __init__(self, filters, w_init, bn_init):
        super(ResidualLayerStage2, self).__init__()
        self.filters = filters
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, input_shape):
        self.conv2d_1 = Conv2D(filters=self.filters, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=self.w_init)
        self.bn_1 = BatchNormalization(gamma_initializer=self.bn_init)

        self.conv2d_2 = Conv2D(filters=self.filters, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=self.w_init)
        self.bn_2 = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        inputs = x

        res = self.conv2d_1(x)
        res = self.bn_1(res, training=training)
        res = tf.nn.relu(res)

        res = self.conv2d_2(res)
        res = self.bn_2(res, training=training)

        res = tf.add(inputs, res)
        return tf.nn.relu(res)
