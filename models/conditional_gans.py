from models.discriminators import Discriminator
from models.generators import Generator
from tensorflow.keras import Model


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
