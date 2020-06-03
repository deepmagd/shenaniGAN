from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose


class DeconvBlock(layers.Layer):

    def __init__(self, filters, w_init, bn_init, activation=None):
        super(DeconvBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, inout_shape):
        self.deconv2d = Conv2DTranspose(self.filters, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init)
        self.conv2d = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=self.w_init)
        self.bn = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        x = self.deconv2d(x)
        x = self.conv2d(x)
        x = self.bn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ConvBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, padding, w_init, bn_init, activation=None):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.w_init = w_init
        self.bn_init = bn_init

    def build(self, input_shape):
        self.conv2d = Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, strides=self.strides,
            padding=self.padding, kernel_initializer=self.w_init
        )
        self.bn = BatchNormalization(gamma_initializer=self.bn_init)

    def call(self, x, training=True):
        x = self.conv2d(x)
        x = self.bn(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x
