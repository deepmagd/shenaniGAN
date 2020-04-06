import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten

from models.discriminators import (Discriminator, DiscriminatorStage1,
                                   DiscriminatorStage2)
from models.generators import Generator, GeneratorStage1, GeneratorStage2
from utils.utils import product_list, sample_normal


class ConditionalGAN(Model):
    """ Definition for a generalisable conditional GAN """
    def __init__(self, generator=None, discriminator=None, **kwargs):
        super().__init__()
        self.generator = generator if generator is not None else Generator(
            img_size=kwargs.get("img_size"),
            num_latent_dims=kwargs.get("num_latent_dims"),
            kernel_size=kwargs.get("kernel_size"),
            num_filters=kwargs.get("num_filters"),
            reshape_dims=kwargs.get("reshape_dims")
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
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims):

        generator = GeneratorStage1(
            img_size=img_size,
            num_latent_dims=num_latent_dims,
            kernel_size=kernel_size,
            num_filters=num_filters,
            reshape_dims=reshape_dims
        )

        discriminator = DiscriminatorStage1(
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

        self.flatten = Flatten()
        self.dense_mean = Dense(units=128, activation='relu') # NOTE change units value to variable
        self.dense_sigma = Dense(units=128, activation='relu') # NOTE change units value to variable

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
        x = tf.reduce_mean(embedding, 1)
        # x = self.flatten(x)
        mean = self.dense_mean(x)
        sigma = self.dense_sigma(x)
        return mean, sigma

    def conditional_augmentation(self, embedding):
        """ Perform conditional augmentation by sampling embedding.
            Arguments
            embedding : Tensor
                text embedding. Shape (batch_size, feature_size, embedding_size)

            Returns
            Tensor
                sampled embedding. Shape (batch_size, reshape_dims/2)
        """
        mean, sigma = self.generate_conditionals(embedding)
        smoothed_embedding = sample_normal(mean, sigma)
        return smoothed_embedding, mean, sigma


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
