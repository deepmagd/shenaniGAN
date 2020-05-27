import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense

from shenanigan.layers import ConvBlock, DeconvBlock
from shenanigan.models import ConditionalGAN, Discriminator, Generator
from shenanigan.models.stackgan.layers import (ConditionalAugmentation,
                                               ResidualLayer)
from shenanigan.models.stackgan.stage2.layers import ResidualLayerStage2


class StackGAN2(ConditionalGAN):
    """ Definition for the stage 2 StackGAN """
    def __init__(self, img_size, lr_g, lr_d, conditional_emb_size, w_init, bn_init):

        generator = GeneratorStage2(
            img_size=img_size,
            lr=lr_g,
            conditional_emb_size=conditional_emb_size,
            w_init=w_init,
            bn_init=bn_init
        )

        discriminator = DiscriminatorStage2(
            img_size=img_size,
            lr=lr_d,
            conditional_emb_size=conditional_emb_size,
            w_init=w_init,
            bn_init=bn_init
        )

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            img_size=img_size
        )

class GeneratorStage2(Generator):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                reshape_dims : tuple or list TODO: actually use
                    [91, 125, 128]
                lr : float
        """
        super().__init__(img_size, lr, conditional_emb_size, w_init, bn_init)
        self.num_output_channels = self.img_size[0]
        self.conditional_emb_size = conditional_emb_size
        assert self.num_output_channels == 3 or self.num_output_channels == 1, \
            f'The number of output channels must be 2 or 1. Found {self.num_output_channels}'

    def build(self, input_shape):

        # NOTE in authors implementation they do not use w_init in stage 2
        self.conv2d_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), kernel_initializer=self.w_init)

        self.conv_block_1 = ConvBlock(filters=128*2,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      w_init=self.w_init,
                                      bn_init=self.bn_init,
                                      activation=tf.nn.relu
                                      )
        self.conv_block_2 = ConvBlock(filters=128*4,
                                      kernel_size=(4, 4),
                                      strides=(2, 2),
                                      padding='same',
                                      w_init=self.w_init,
                                      bn_init=self.bn_init,
                                      activation=tf.nn.relu
                                      )

        self.conditional_augmentation = ConditionalAugmentation(self.conditional_emb_size, self.w_init)

        self.conv_block_3 = ConvBlock(filters=128*4,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      w_init=self.w_init,
                                      bn_init=self.bn_init,
                                      activation=tf.nn.relu
                                      )

        self.res_block_1 = ResidualLayerStage2(filters=128*4, w_init=self.w_init, bn_init=self.bn_init)
        self.res_block_2 = ResidualLayerStage2(filters=128*4, w_init=self.w_init, bn_init=self.bn_init)
        self.res_block_3 = ResidualLayerStage2(filters=128*4, w_init=self.w_init, bn_init=self.bn_init)
        self.res_block_4 = ResidualLayerStage2(filters=128*4, w_init=self.w_init, bn_init=self.bn_init)

        self.deconv_block_1 = DeconvBlock(128*2, self.w_init, self.bn_init, activation=tf.nn.relu)
        self.deconv_block_2 = DeconvBlock(128, self.w_init, self.bn_init, activation=tf.nn.relu)
        self.deconv_block_3 = DeconvBlock(128//2, self.w_init, self.bn_init, activation=tf.nn.relu)
        self.deconv_block_4 = DeconvBlock(128//4, self.w_init, self.bn_init, activation=tf.nn.relu)

        self.conv2d_2 = Conv2D(filters=self.num_output_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.tanh = Activation('tanh')

    def call(self, inputs, training=True):
        generated_image, embedding = inputs

        x = self.conv2d_1(generated_image)
        x = tf.nn.relu(x)

        x = self.conv_block_1(x, training=training)
        x = self.conv_block_2(x, training=training)

        smoothed_embedding, mean, log_sigma = self.conditional_augmentation(embedding)

        smoothed_embedding = tf.expand_dims(tf.expand_dims(smoothed_embedding, 1), 1)
        smoothed_embedding = tf.tile(smoothed_embedding, [1, 16, 16, 1])
        x = tf.concat([x, smoothed_embedding], 3)

        x = self.conv_block_3(x, training=training)

        x = self.res_block_1(x, training=training)
        x = self.res_block_2(x, training=training)
        x = self.res_block_3(x, training=training)
        x = self.res_block_4(x, training=training)

        x = self.deconv_block_1(x, training=training)
        x = self.deconv_block_2(x, training=training)
        x = self.deconv_block_3(x, training=training)
        x = self.deconv_block_4(x, training=training)

        x = self.conv2d_2(x)
        x = self.tanh(x)

        return x, mean, log_sigma

    def loss(self, predictions_on_fake, mean, log_sigma):
        """ Only calculate the loss based on the discriminator
            predictions for the images generated by this model.
        """
        kl_coeff = 2
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(predictions_on_fake), logits=predictions_on_fake
            ))
        kl_loss = self.kl_loss(mean, log_sigma)
        loss = loss + kl_coeff * kl_loss
        return loss, kl_loss

    def kl_loss(self, mean, log_sigma):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.math.square(mean))
        loss = tf.reduce_mean(loss)
        return loss


class DiscriminatorStage2(Discriminator):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size, lr, conditional_emb_size, w_init, bn_init):
        """ Initialise a Generator instance.
            TODO: Deal with this parameters and make it more logical
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                lr : float
        """
        super().__init__(img_size, lr, w_init, bn_init)
        self.d_dim = 64
        self.conditional_emb_size = conditional_emb_size

    def build(self, input_size):
        activation = lambda l: tf.nn.leaky_relu(l, alpha=0.2)

        self.conv_1 = Conv2D(filters=self.d_dim, kernel_size=(4, 4), strides=(2, 2), padding='same',
                             kernel_initializer=self.w_init, use_bias=False)

        self.conv_block_2 = ConvBlock(filters=self.d_dim*2, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_3 = ConvBlock(filters=self.d_dim*4, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_4 = ConvBlock(filters=self.d_dim*8, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_5 = ConvBlock(filters=self.d_dim*16, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_6 = ConvBlock(filters=self.d_dim*32, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_7 = ConvBlock(filters=self.d_dim*16, kernel_size=(4, 4), strides=(1, 1), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        self.conv_block_8 = ConvBlock(filters=self.d_dim*8, kernel_size=(4, 4), strides=(1, 1), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init)

        self.res_block = ResidualLayer(self.d_dim*2, self.d_dim*8, self.w_init, self.bn_init)

        self.dense_embed = Dense(units=self.conditional_emb_size)

        self.conv_block_9 = ConvBlock(filters=self.d_dim*8, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                      w_init=self.w_init, bn_init=self.bn_init, activation=activation)

        # (4, 4) == 256/16
        self.conv_2 = Conv2D(filters=1, kernel_size=(4, 4), strides=(4, 4), padding='same',
                             kernel_initializer=self.w_init, use_bias=False)

    def call(self, inputs, training=True):
        images, embedding = inputs

        x = self.conv_1(images)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv_block_2(x, training=training)
        x = self.conv_block_3(x, training=training)
        x = self.conv_block_4(x, training=training)
        x = self.conv_block_5(x, training=training)
        x = self.conv_block_6(x, training=training)
        x = self.conv_block_7(x, training=training)
        x = self.conv_block_8(x, training=training)

        res = self.res_block(x, training=training)
        x = tf.add(x, res)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        reduced_embedding = self.dense_embed(embedding)
        reduced_embedding = tf.nn.leaky_relu(reduced_embedding, alpha=0.2)
        reduced_embedding = tf.expand_dims(tf.expand_dims(reduced_embedding, 1), 1)
        reduced_embedding = tf.tile(reduced_embedding, [1, 4, 4, 1])
        x = tf.concat([x, reduced_embedding], 3)

        x = self.conv_block_9(x, training=training)
        x = self.conv_2(x)

        return x

    def loss(self, predictions_on_real, predictions_on_wrong, predictions_on_fake):
        """ Calculate the loss for the predictions made on real and fake images.
                Arguments:
                predictions_on_real : Tensor
                predictions_on_fake : Tensor
        """
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(predictions_on_real), logits=predictions_on_real))
        wrong_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions_on_wrong), logits=predictions_on_wrong))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(predictions_on_fake), logits=predictions_on_fake))
        total_loss = real_loss + (wrong_loss + fake_loss) / 2
        return total_loss
