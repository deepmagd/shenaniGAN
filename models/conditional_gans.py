from models.discriminators import Discriminator
from models.generators import Generator
from tensorflow.keras import Model


class ConditionalGAN(Model):
    """ Definition for a generalisable conditional GAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims, num_image_channels):
        super().__init__()
        self.generator = Generator(
            img_size, num_latent_dims, kernel_size, num_filters,
            reshape_dims, num_image_channels
        )
        self.discriminator = Discriminator(
            img_size, kernel_size, num_filters, num_image_channels
        )


class StackGAN1(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims, num_image_channels):
        super().__init__(img_size, num_latent_dims, kernel_size,
                         num_filters, reshape_dims, num_image_channels)


class StackGAN2(ConditionalGAN):
    """ Definition for the stage 1 StackGAN """
    def __init__(self, img_size, num_latent_dims, kernel_size,
                 num_filters, reshape_dims, num_image_channels):
        super().__init__(img_size, num_latent_dims, kernel_size,
                         num_filters, reshape_dims, num_image_channels)
