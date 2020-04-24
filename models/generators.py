import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, LeakyReLU,
                                     Dense, Reshape, Conv2DTranspose,
                                     ReLU, Flatten)
from tensorflow.keras.activations import tanh
from utils.utils import sample_normal


class Generator(Model):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, num_latent_dims, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:'
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                num_latent_dims : int
                    Dimensionality of latent input.
                lr: float
                    Learning rate
                conditional_emb_size: Tensor
                    text embedding. Shape (batch_size, feature_size, embedding_size)
        """
        super().__init__()
        self.img_size = img_size
        self.num_latent_dims = num_latent_dims

        # Optimiser
        self.optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        # Weight Initialisation Parameters
        self.w_init = w_init
        self.bn_init = bn_init

        # Conditional Layers
        self.flatten = Flatten()
        self.dense_mean = Dense(units=conditional_emb_size, activation='relu')
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.dense_sigma = Dense(units=conditional_emb_size, activation='relu')
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
        smoothed_embedding = sample_normal(mean, log_sigma)
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
        # NOTE embedding is dim (batch, 10, 1024) where 10 is different samples for same image. Options are to either
        # flatten all features or average across embeddings
        mean = self.leaky_relu1(self.dense_mean(embedding))
        log_sigma = self.leaky_relu2(self.dense_sigma(embedding))
        return mean, log_sigma

class GeneratorStage1(Generator):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, num_latent_dims, kernel_size, num_filters,
                 reshape_dims, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                num_latent_dims : int
                    Dimensionality of latent input.
                kernel_size : tuple
                    (height, width)
                reshape_dims : tuple or list TODO: actually use
                    [91, 125, 128]
                lr : float
        """
        super().__init__(img_size, num_latent_dims, lr, conditional_emb_size, w_init, bn_init)
        num_output_channels = self.img_size[0]
        assert num_output_channels == 3 or num_output_channels == 1, \
            f'The number of output channels must be 2 or 1. Found {num_output_channels}'

        self.dense1 = Dense(units=128*8*4*4, kernel_initializer=self.w_init)
        self.bn1 = BatchNormalization(gamma_initializer=self.bn_init)
        self.reshape_layer = Reshape([4, 4, 128*8])
        self.relu1 = ReLU()

        self.deconv2d1 = Conv2DTranspose(
            128*4, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init
        )
        self.conv1 = Conv2D(
            filters=128*4, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=self.w_init
        )
        self.bn2 = BatchNormalization(gamma_initializer=self.bn_init)
        self.relu2 = ReLU()

        self.deconv2d2 = Conv2DTranspose(
            128*2, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init
        )
        self.conv2 = Conv2D(
            filters=128*2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=self.w_init
        )
        self.bn3 = BatchNormalization(gamma_initializer=self.bn_init)
        self.relu3 = ReLU()

        self.deconv2d3 = Conv2DTranspose(
            128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init
        )
        self.conv3 = Conv2D(
            filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=self.w_init
        )
        self.bn4 = BatchNormalization(gamma_initializer=self.bn_init)
        self.relu4 = ReLU()

        self.deconv2d4 = Conv2DTranspose(
            num_output_channels, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=self.w_init
        )
        self.conv4 = Conv2D(
            filters=num_output_channels, kernel_size=(3, 3), strides=(1, 1),
            padding='same', kernel_initializer=self.w_init
        )

    @tf.function
    def call(self, embedding, noise):
        smoothed_embedding, mean, log_sigma = self.conditional_augmentation(embedding)
        noisy_embedding = tf.concat([smoothed_embedding, noise], 1)

        x = self.dense1(noisy_embedding)
        x = self.bn1(x)
        x = self.reshape_layer(x)
        x = self.relu1(x)

        x = self.deconv2d1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.deconv2d2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.deconv2d3(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.deconv2d4(x)
        x = self.conv4(x)

        x = tanh(x)

        return x, mean, log_sigma

    def loss(self, predictions_on_fake, mean, log_sigma):
        """ Only calculate the loss based on the discriminator
            predictions for the images generated by this model.
        """
        kl_coeff = 1
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(predictions_on_fake),
            logits=predictions_on_fake
        )
        loss = tf.reduce_mean(loss) + kl_coeff * self.kl_loss(mean, log_sigma)
        return loss

    def kl_loss(self, mean, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mean))
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
