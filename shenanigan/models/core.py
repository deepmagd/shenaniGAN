import tensorflow as tf
from tensorflow.keras import Model


class ConditionalGAN(Model):
    """ Definition for a generalisable conditional GAN """

    def __init__(self, generator, discriminator, **kwargs):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator


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
        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        # Weight Initialisation Parameters
        self.w_init = w_init
        self.bn_init = bn_init


class Discriminator(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """

    def __init__(self, img_size, lr, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()
        self.img_size = img_size

        # Weight Initialisation Parameters
        self.w_init = w_init
        self.bn_init = bn_init

        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
