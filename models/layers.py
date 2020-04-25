from tensorflow.keras import layers
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, LeakyReLU, ReLU)


class ResidualLayer(layers.Layer):

    def __init__(self, filters_in, filters_out, w_init, bn_init):
        super(ResidualLayer, self).__init__()
        self.conv2d_1 = Conv2D(filters=filters_in, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=w_init)
        self.bn_1 = BatchNormalization(gamma_initializer=bn_init)
        self.relu_1 = ReLU()

        self.conv2d_2 = Conv2D(filters=filters_in, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=w_init)
        self.bn_2 = BatchNormalization(gamma_initializer=bn_init)
        self.relu_2 = ReLU()

        self.conv2d_3 = Conv2D(filters=filters_out, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=w_init)
        self.bn_3 = BatchNormalization(gamma_initializer=bn_init)

    def call(self, x):
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        return self.bn_3(x)

class DeconvBlock(layers.Layer):

    def __init__(self, filters, w_init, bn_init, activation=False):
        super(DeconvBlock, self).__init__()
        self.activation = activation
        self.deconv2d = Conv2DTranspose(filters, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=w_init)
        self.conv2d = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=w_init)
        self.bn = BatchNormalization(gamma_initializer=bn_init)
        self.relu = ReLU()

    def call(self, x):
        x = self.deconv2d(x)
        x = self.conv2d(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding, w_init, bn_init, activation=False):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=w_init)
        self.bn = BatchNormalization(gamma_initializer=bn_init)
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x
