import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense,
                                     LeakyReLU, ReLU, Reshape, Activation)

from models.layers import DeconvBlock, ResidualLayer


class Generator(Model):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:'
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                lr: float
                    Learning rate
                conditional_emb_size: Tensor
                    text embedding. Shape (batch_size, feature_size, embedding_size)
        """
        super().__init__()
        self.img_size = img_size
        self.optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        # Weight Initialisation Parameters
        self.w_init = w_init
        self.bn_init = bn_init

        self.dense_mean = Dense(units=conditional_emb_size, kernel_initializer=self.w_init)
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.dense_sigma = Dense(units=conditional_emb_size, kernel_initializer=self.w_init)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)

    @tf.function
    def call(self, x):
        pass

    def conditional_augmentation(self, embedding):
        """ Perform conditional augmentation by sampling normal distribution generated
            from the embedding.
            Arguments
            embedding : Tensor
                text embedding. Shape (batch_size, feature_size, embedding_size)

            Returns
            Tensor
                sampled embedding. Shape (batch_size, reshape_dims/2)
        """
        mean, log_sigma = self.generate_conditionals(embedding)
        epsilon = tf.random.truncated_normal(tf.shape(mean))
        stddev = tf.math.exp(log_sigma)
        smoothed_embedding = mean + stddev * epsilon
        return smoothed_embedding, mean, log_sigma

    def generate_conditionals(self, embedding):
        """ Generate distribution for text embedding.
            Arguments
            embedding : Tensor
                text embedding. Shape (batch_size, feature_size, embedding_size)

            Returns
            Tensor
                learnt mean of embedding distribution. Shape (batch_size, reshape_dims/2)
            Tensor
                learnt log variance of embedding distribution. Shape (batch_size, reshape_dims/2)

        """
        mean = self.leaky_relu1(self.dense_mean(embedding))
        log_sigma = self.leaky_relu2(self.dense_sigma(embedding))
        return mean, log_sigma


class GeneratorStage1(Generator):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, kernel_size, num_filters,
                 reshape_dims, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                kernel_size : tuple
                    (height, width)
                reshape_dims : tuple or list TODO: actually use
                    [91, 125, 128]
                lr : float
        """
        super().__init__(img_size, lr, conditional_emb_size, w_init, bn_init)
        num_output_channels = self.img_size[0]
        assert num_output_channels == 3 or num_output_channels == 1, \
            f'The number of output channels must be 2 or 1. Found {num_output_channels}'

        self.dense_1 = Dense(units=128*8*4*4, kernel_initializer=self.w_init)
        self.bn_1 = BatchNormalization(gamma_initializer=self.bn_init)
        self.reshape_layer = Reshape([4, 4, 128*8])

        self.res_block_1 = ResidualLayer(128*2, 128*8, self.w_init, self.bn_init)
        self.relu_1 = ReLU()

        self.deconv_block_1 = DeconvBlock(128*4, self.w_init, self.bn_init, activation=False)

        self.res_block_2 = ResidualLayer(128, 128*4, self.w_init, self.bn_init)
        self.relu_2 = ReLU()

        self.deconv_block_2 = DeconvBlock(128*2, self.w_init, self.bn_init, activation=True)
        self.deconv_block_3 = DeconvBlock(128, self.w_init, self.bn_init, activation=True)

        self.deconv2d_4 = Conv2DTranspose(
            num_output_channels, kernel_size=(4, 4), strides=(2, 2),
            padding='same', kernel_initializer=self.w_init
        )
        self.conv2d_4 = Conv2D(
            filters=num_output_channels, kernel_size=(3, 3), strides=(1, 1),
            padding='same', kernel_initializer=self.w_init, use_bias=False
        )

        self.tanh = Activation('tanh')

    @tf.function
    def call(self, embedding, noise, training=True):
        smoothed_embedding, mean, log_sigma = self.conditional_augmentation(embedding)
        noisy_embedding = tf.concat([noise, smoothed_embedding], 1)

        x = self.dense_1(noisy_embedding)
        x = self.bn_1(x, training=training)
        x = self.reshape_layer(x)

        res_1 = self.res_block_1(x, training=training)
        x = tf.add(x, res_1)
        x = self.relu_1(x)

        x = self.deconv_block_1(x, training=training)

        res_2 = self.res_block_2(x, training=training)
        x = tf.add(x, res_2)
        x = self.relu_2(x)

        x = self.deconv_block_2(x, training=training)
        x = self.deconv_block_3(x, training=training)

        x = self.deconv2d_4(x)
        x = self.conv2d_4(x)

        x = self.tanh(x)

        return x, mean, log_sigma

    def loss(self, predictions_on_fake, mean, log_sigma):
        """ Only calculate the loss based on the discriminator
            predictions for the images generated by this model.
        """
        kl_coeff = 1
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(predictions_on_fake), logits=predictions_on_fake
            ))
        loss = loss + kl_coeff * self.kl_loss(mean, log_sigma)
        return loss

    def kl_loss(self, mean, log_sigma):
        loss = .5 * (-log_sigma - 1 + tf.exp(2. * log_sigma) + tf.math.square(mean))
        loss = tf.reduce_mean(loss)
        return loss


class GeneratorStage2(Model):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, num_latent_dims, kernel_size, num_filters,
                 reshape_dims):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                num_latent_dims : int
                    Dimensionality of latent input.
                kernel_size : tuple
                    (height, width)
                reshape_dims : tuple or list
                    [91, 125, 128]
        """
        super().__init__()
        pass

    @tf.function
    def call(self, noise, embedding):
        pass

    def loss(self, predictions_on_fake):
        pass
