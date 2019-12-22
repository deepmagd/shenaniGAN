import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LeakyReLU


class Discriminator(Model):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size, kernel_size, num_filters, num_channels=3):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super().__init__()
        self.img_size = img_size
        self.xent_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimiser = tf.keras.optimizers.Adam(1e-4)
        # TODO: Add the correct layers as per the paper
        self.leaky_relu_1 = LeakyReLU()
        self.conv1 = Conv2D(filters=num_channels, kernel_size=kernel_size, activation='relu', padding='same')
        self.leaky_relu_2 = LeakyReLU()
        self.conv2 = Conv2D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding="same")
        self.flatten = Flatten()
        self.dense1 = Dense(units=32, activation='relu')
        self.dense2 = Dense(units=1, activation='sigmoid')

    def call(self, images):
        x = self.leaky_relu_1(images)
        x = self.conv1(x)
        x = self.leaky_relu_2(x)
        x = self.conv2(x)
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