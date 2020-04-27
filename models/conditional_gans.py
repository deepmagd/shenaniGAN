import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU

from models.discriminators import (Discriminator, DiscriminatorStage1,
                                   DiscriminatorStage2)
from models.generators import Generator, GeneratorStage1, GeneratorStage2


class ConditionalGAN(Model):
    """ Definition for a generalisable conditional GAN """
    def __init__(self, generator=None, discriminator=None, **kwargs):
        super().__init__()
        self.generator = generator if generator is not None else Generator(
            img_size=kwargs.get("img_size"),
            num_latent_dims=kwargs.get("num_latent_dims"),
            lr=kwargs.get("lr"),
            conditional_emb_size=kwargs.get("conditional_emb_size")
        )
        self.discriminator = discriminator if discriminator is not None else Discriminator(
            img_size=kwargs.get("img_size"),
            kernel_size=kwargs.get("kernel_size"),
            num_filters=kwargs.get("num_filters")
        )

    def generate_images(self, num_images):
        """ Generate a number of fake images from
            random noise from a normal distribution.
            Arguments:
                num_images: int
                    Number of fake images to return
        """
        image_list = []
        for _ in range(num_images):
            image = self.generate_image()
            image_list.append(image)
        return image_list

    def generate_image(self, noise=None):
        """ Generate a fake image from a latenet space
            of random noise from a normal distribution.
            If the `noise` parameter is not None, then
            use the provided tensor.
            Arguments:
                noise: The latent noise to convert
                       to an image.
        """
        if noise is None:
            noise = tf.random.normal([1, self.generator.num_latent_dims])
        return self.generator(noise)

    def classify_images(self, images):
        """ Given a list of images, return the prediction
            for whether it is fake or real.
            Arguments:
                images: list of images (TODO: later add image & text pairs)
        """
        predictions = []
        for image in images:
            prediction = self.discriminator(image)
            predictions.append(prediction)
        return predictions


class StackGAN1(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, kernel_size, num_filters, reshape_dims,
                 lr_g, lr_d, conditional_emb_size, w_init, bn_init):

        generator = GeneratorStage1(
            img_size=img_size,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims,
            lr=lr_g,
            conditional_emb_size=conditional_emb_size,
            w_init=w_init,
            bn_init=bn_init
        )

        discriminator = DiscriminatorStage1(
            img_size=img_size,
            kernel_size=kernel_size,
            num_filters=num_filters,
            lr=lr_d,
            w_init=w_init,
            bn_init=bn_init
        )

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            img_size=img_size,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims
        )


class StackGAN2(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims):

        generator = GeneratorStage2(
            img_size=img_size,
            num_latent_dims=num_latent_dims,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims
        )

        discriminator = DiscriminatorStage2(
            img_size=img_size,
            kernel_size=kernel_size,
            num_filters=num_filters
        )

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            img_size=img_size,
            num_latent_dims=num_latent_dims,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims
        )

    @tf.function
    def call(self, embedding, z):
        """
        """
        raise NotImplementedError
