import tensorflow as tf
from tensorflow.keras import Model


class Discriminator(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size):
        """ Initialise a Generator instance.
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()
        self.img_size = img_size
        self.xent_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # TODO: Add layers

    def call(self, images):
        pass

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