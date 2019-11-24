class Discriminator(object):
    """ The definition for a network which
        classifies inputs as fake or genuine.
    """
    def __init__(self, img_size):
        """ Initialise a Generator instance.
                Arguments:
                img_size : tuple of ints
                    Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        self.img_size = img_size