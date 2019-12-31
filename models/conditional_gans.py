from models.discriminators import Discriminator
from models.generators import Generator
from tensorflow.keras import Model
import tensorflow as tf


class ConditionalGAN(Model):
    """ Definition for a generalisable conditional GAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims):
        super().__init__()
        self.generator = Generator(
            img_size=img_size,
            num_latent_dims=num_latent_dims,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims
        )
        self.discriminator = Discriminator(
            img_size=img_size,
            kernel_size=kernel_size,
            num_filters=num_filters
        )

    def generate_images(self, num_images):
        """ Generate a number of fake images from
            random noise from a normal distribution.
            Arguments:
                num_images: int
                    Number of fake images to return
        """
        image_list = []
        for image_idx in range(num_images):
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
                images: list of images (and later image & text pairs)
        """
        predictions = []
        for image in images:
            prediction = self.discriminator(image)
            predictions.append(prediction)
        return predictions


class StackGAN1(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims):
        super().__init__(img_size, num_latent_dims, kernel_size,
                         num_filters, reshape_dims)


class StackGAN2(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims):
        super().__init__(img_size, num_latent_dims, kernel_size,
                         num_filters, reshape_dims)
