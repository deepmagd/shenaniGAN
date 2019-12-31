import glob
import json
import numpy as np
import os
import pathlib
from random import randint
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
        download_flowers()
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

def download_flowers():
    """ Download the flowers dataset """
    FLOWERS_DATASET_URL = "www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    print('Downloading the flowers dataset from: {}'.format(FLOWERS_DATASET_URL))

    flowers_download_loc = pathlib.Path('data/flowers/flowers.tgz')
    urllib.request.urlretrieve(FLOWERS_DATASET_URL, flowers_download_loc)
    tar = tarfile.open(flowers_download_loc, "r:gz")
    tar.extractall("data/flowers/images")
    os.remove(flowers_download_loc)

    DATA_SPLITS_URL = "www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    data_splits_download_loc = pathlib.Path('data/flowers/setid.mat')
    urllib.request.urlretrieve(DATA_SPLITS_URL, data_splits_download_loc)

    IMAGE_LABELS_URL = "www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    image_labels_download_loc = pathlib.Path('data/flowers/imagelabels.mat')
    urllib.request.urlretrieve(IMAGE_LABELS_URL, image_labels_download_loc)

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

def product_list(num_list):
    """ A helper function to simply find the
        product of all elements in the list.
    """
    product = 1
    for dim in num_list:
        product *= dim
    return product

def mkdir(directory):
    """ Create directory if it does not exist. """
    try:
        os.makedirs(directory)
    except OSError:
        print("Directory {} already exists.".format(directory))

def save_options(options, save_dir):
    """ Save all options to JSON file.
        Arguments:
            options: An object from argparse
            save_dir: String location to save the options
    """
    opt_dict = {}
    for option in vars(options):
        opt_dict[option] = getattr(options, option)

    opts_file_path = os.path.join(save_dir, 'opts.json')
    with open(opts_file_path, 'w') as opt_file:
        json.dump(opt_dict, opt_file)

def sample_real_images(num_images, dataset_name):
    """ Randomly sample images (with replacement) from all
        available images in the data directory.
        Arguments:
            num_images: int
                The number of images to sample from the dataset
            data_dir: str
                The directory where the images are saved to disk
    """
    dataset = get_dataset(dataset_name)
    sampled_image_paths = sample_image_paths(dataset.directory, num_images)
    sampled_images = get_images_from_paths(sampled_image_paths, dataset)
    return sampled_images

def sample_image_paths(data_dir, num_paths):
    """ Randomly sample image paths from the provided data directory """
    for root, dirs, names in os.walk(data_dir):
        image_paths = [os.path.join(root, name) for name in names]

    sampled_image_paths = []
    for _ in range(num_paths):
        sample_idx = randint(0, len(image_paths) - 1)
        sampled_image_paths.append(image_paths[sample_idx])
    return sampled_image_paths

def get_images_from_paths(sampled_image_paths, dataset):
    """ Given a list of paths to images, and the dataset object
        to which they belong, load all images into a list.
    """
    images = []
    for image_path in sampled_image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=NUM_COLOUR_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # NOTE: Really not sure why the height and the width need to be
        #       this way around here but above they are reversed
        image = tf.image.resize(image, [dataset.height, dataset.width])
        images.append(tf.expand_dims(image, axis=0))
    return images
