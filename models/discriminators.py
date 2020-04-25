import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense,
                                     Flatten, LeakyReLU)
from tensorflow.keras.activations import sigmoid
from models.layers import ResidualLayer, ConvBlock

class Discriminator(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size, kernel_size, num_filters):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()
        self.img_size = img_size
        num_channels = self.img_size[0]
        self.xent_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimiser = tf.keras.optimizers.Adam(1e-4)
        # TODO: Add the correct layers as per the paper
        self.leaky_relu_1 = LeakyReLU()
        self.conv1 = Conv2D(filters=num_channels, kernel_size=kernel_size, padding='same')
        self.leaky_relu_2 = LeakyReLU()
        self.conv2 = Conv2D(filters=num_filters, kernel_size=kernel_size, padding="same")
        self.bn1 = BatchNormalization()
        self.conv3 = Conv1D(filters=num_filters, kernel_size=1, padding="same")
        self.flatten = Flatten()
        self.dense1 = Dense(units=32, activation='relu')
        self.dense2 = Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, images, embedding):
        x = self.conv1(images)
        x = self.leaky_relu_1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu_2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def loss(self, predictions_on_real, predictions_on_fake):
        """ Calculate the loss for the predictions made on real and fake images.
                Arguments:
                predictions_on_real : Tensor
                predictions_on_fake : Tensor
        """
        real_loss = self.xent_loss_fn(tf.ones_like(predictions_on_real), predictions_on_real)
        fake_loss = self.xent_loss_fn(tf.zeros_like(predictions_on_fake), predictions_on_fake)
        total_loss = real_loss + fake_loss
        return total_loss


class DiscriminatorStage1(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size, kernel_size, num_filters, lr):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                lr : float
        """
        super().__init__()

        self.w_init = tf.random_normal_initializer(stddev=0.02)
        self.bn_init = tf.random_normal_initializer(1., 0.02)

        self.img_size = img_size
        num_channels = self.img_size[0]
        # self.xent_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        # TODO: Add the correct layers as per the paper
        self.conv_1 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init)
        self.leaky_relu_1 = LeakyReLU(alpha=0.2)

        self.conv_block_1 = ConvBlock(filters=64*2, kernel_size=(4, 4), strides=(2, 2), padding='same', w_init=self.w_init, bn_init=self.bn_init, activation=True)
        self.conv_block_2 = ConvBlock(filters=64*4, kernel_size=(4, 4), strides=(2, 2), padding='same', w_init=self.w_init, bn_init=self.bn_init, activation=True)
        self.conv_block_3 = ConvBlock(filters=64*8, kernel_size=(4, 4), strides=(2, 2), padding='same', w_init=self.w_init, bn_init=self.bn_init, activation=False)

        self.res_block = ResidualLayer(64*2, 64*8, self.w_init, self.bn_init)
        self.leaky_relu_2 = LeakyReLU(alpha=0.2)

        self.dense_embed = Dense(units=128)
        self.leaky_relu_3 = LeakyReLU(alpha=0.2)

        self.conv_block_4 = ConvBlock(filters=64*8, kernel_size=(1, 1), strides=(1, 1), padding='valid', w_init=self.w_init, bn_init=self.bn_init, activation=True)

        # (4, 4) == 64/16
        self.conv_2 = Conv2D(filters=1, kernel_size=(4, 4), strides=(4, 4), padding="valid", kernel_initializer=self.w_init)

    @tf.function
    def call(self, images, embedding):

        x = self.conv_1(images)
        x = self.leaky_relu_1(x)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        res = self.res_block(x)
        x = tf.add(x, res)
        x = self.leaky_relu_2(x)

        reduced_embedding = self.dense_embed(embedding)
        reduced_embedding = self.leaky_relu_3(reduced_embedding)
        reduced_embedding = tf.expand_dims(tf.expand_dims(reduced_embedding, 1), 1)
        reduced_embedding = tf.tile(reduced_embedding, [1, 4, 4, 1])
        x = tf.concat([x, reduced_embedding], 3)

        x = self.conv_block_4(x)
        x = self.conv_2(x)

        return x

    def loss(self, predictions_on_real, predictions_on_fake):
        """ Calculate the loss for the predictions made on real and fake images.
                Arguments:
                predictions_on_real : Tensor
                predictions_on_fake : Tensor
        """
        real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictions_on_real), logits=predictions_on_real)
        real_loss = tf.reduce_mean(real_loss)
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions_on_fake), logits=predictions_on_fake)
        fake_loss = tf.reduce_mean(fake_loss)
        # real_loss = self.xent_loss_fn(tf.ones_like(predictions_on_real), predictions_on_real)
        # fake_loss = self.xent_loss_fn(tf.zeros_like(predictions_on_fake), predictions_on_fake)
        total_loss = real_loss + fake_loss
        return total_loss

class DiscriminatorStage2(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size, kernel_size, num_filters):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()
        pass

    @tf.function
    def call(self, images, embedding):
        pass

    def loss(self, predictions_on_real, predictions_on_fake):
        pass
