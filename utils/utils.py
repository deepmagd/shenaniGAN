# from ast import eval
import glob
import numpy as np
import os
import pathlib
import tarfile
import tensorflow as tf
import urllib.request
import yaml


AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASETS_DICT = {
    "birds": "BirdsDataset",
    "flowers": "FlowersDataset",
    "xrays": "XRaysDataset"
}
DATASETS = list(DATASETS_DICT.keys())
NUM_COLOUR_CHANNELS = 3


class StackedGANDataset(object):
    """ Base class for all datasets """
    def __init__(self):
        self.directory = None
        self.image_label_pairs = None
        self.classes = None
        self.width = None
        self.height = None

    def get_image_label_pairs(self):
        image_paths = tf.data.Dataset.list_files(str(self.directory/'*/*'))
        self.image_label_pairs = image_paths.map(self.process_path, num_parallel_calls=AUTOTUNE)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def get_label(self, file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] == self.classes

    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=NUM_COLOUR_CHANNELS)
        # Convert to floats in the [0,1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Resize the image to the desired size
        return tf.image.resize(img, [self.width, self.height])

class BirdsDataset(StackedGANDataset):
    """ Container for the birds dataset properties """
    def __init__(self):
        super().__init__()
        self.directory = pathlib.Path(os.path.join('data/CUB_200_2011/CUB_200_2011/images'))
        if not os.path.isdir(self.directory):
            download_dataset(dataset='birds')
        self.classes = np.array(
            [item.name for item in self.directory.glob('*') if os.path.isdir(item.name)]
        )
        self.width = 500
        self.height = 364
        self.num_channels = 3
        self.get_image_label_pairs()
    
    def get_dims(self):
        return (self.num_channels, self.height, self.width)

class FlowersDataset(StackedGANDataset):
    """ TODO: Container for the birds dataset properties """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class XRaysDataset(StackedGANDataset):
    """ TODO: Container for the x-rays dataset properties """
    def __init__(self):
        super().__init__()
        raise NotImplementedError

def download_dataset(dataset):
    if dataset == 'birds':
        download_cub()
    elif dataset == 'flowers':
        raise NotImplementedError
    elif dataset == 'xrays':
        raise NotImplementedError
    else:
        raise NotImplementedError

def download_cub():
    """ Download the birds dataset (CUB-200-2011) """
    BIRDS_DATASET_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    print('Downloading CUB dataset from: {}'.format(BIRDS_DATASET_URL))
    download_location = pathlib.Path('data/CUB_200_2011.tgz')
    urllib.request.urlretrieve(BIRDS_DATASET_URL, download_location)
    tar = tarfile.open(download_location, "r:gz")
    tar.extractall("data/CUB_200_2011")
    os.remove(download_location)

def get_dataset(dataset_name):
    """ Get the dataset object which contains information
        about the properties of the dataset
    """
    if dataset_name in DATASETS:
        dataset = DATASETS_DICT[dataset_name]
        print(dataset)
        return eval(dataset)()
    else:
        raise Exception('Invalid dataset name {}.'.format(dataset_name))

def get_default_settings(settings_file='settings.yml'):
    with open(settings_file) as f:
        return yaml.safe_load(f)

def sample_normal(mean, log_var):
    """ Use the reparameterization trick to sample a normal distribution.
        Arguments
        mean : Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        log_var : Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
    """
    std = tf.math.exp(0.5 * log_var)
    epsilon = tf.random.normal(std)
    return mean + std * epsilon
