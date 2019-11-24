class Generator(object):
    """ The definition for a network which
        fabricates images from a noisy distribution.
    """
    def __init__(self, img_size, num_latent_dims):
        """ Initialise a Generator instance.
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
                num_latent_dims : int
                    Dimensionality of latent input.
        """
        self.img_size = img_size
        self.num_latent_dims = num_latent_dims
