import glob
from google_drive_downloader import GoogleDriveDownloader as gdd
import json
from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pathlib
import pickle
from random import randint
import tarfile
import tensorflow as tf
import urllib.request
import yaml


AUTOTUNE = tf.data.experimental.AUTOTUNE
DATASETS_DICT = {
    "birds": "BirdsDataset",
    "birds-with-text": "BirdsWithWordsDataset",
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
        self.num_channels = None

    def get_dims(self):
        return (self.num_channels, self.height, self.width)

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
        img = tf.image.convert_image_dtype(img, data_type=tf.float32)
        # Resize the image to the desired size
        return tf.image.resize(img, [self.width, self.height])

class BirdsDataset(StackedGANDataset):
    """ Container for the birds dataset properties. """
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


class BirdsWithWordsDataset(StackedGANDataset):
    """ Container for the birds dataset which includes word captions """
    def __init__(self):
        """ TODO: Not yet implemented
        """
        super().__init__()
        # The directory to the TFRecords
        # os.path.join('data/CUB_200_2011_with_text/CUB_200_2011/images')
        self.width = 500
        self.height = 364
        self.num_channels = 3

        self.directory = pathlib.Path(
            os.path.join('data/CUB_200_2011_with_text/')
        )
        if not os.path.isdir(self.directory):
            download_dataset(dataset='birds-with-text')
            create_tfrecords(
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images', 'CUB_200_2011', 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                image_dims=(self.height, self.width),
                samples_per_shard=10
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir

        self.classes = None
        self.dataset_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'names': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenFeature([], tf.string),
            'classes': tf.io.FixedLenFeature([], tf.float32)
        }

        self.train_set, self.test_set = self.parse_dataset()

    def parse_dataset(self):
        """ """
        parsed_subsets = []
        for subset in ['train', 'test']:
            subset_paths = get_record_paths(os.path.join(self.directory, subset))
            raw_subset = tf.data.TFRecordDataset(subset_paths)
            parsed_subset = raw_subset.map(self._parse_example)
            parsed_subsets.append(parsed_subset)
        return parsed_subsets[0], parsed_subsets[1]

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using self.dataset_description
        return tf.io.parse_single_example(example_proto, self.dataset_description)

def get_record_paths(root_dir):
    """ """
    record_path_names = []
    for root, dirs, record_names in os.walk(root_dir):
        for record_name in record_names:
            if record_name.endswith('.tfrecord'):
                record_path_name = os.path.join(root, record_name)
                record_path_names.append(record_path_name)
    return record_path_names


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
    elif dataset == 'birds-with-text':
        download_cub(include_text=True)
    elif dataset == 'flowers':
        download_flowers()
    elif dataset == 'xrays':
        raise NotImplementedError
    else:
        raise NotImplementedError

def download_cub(include_text=False):
    """ Download the birds dataset (CUB-200-2011) """
    BIRDS_DATASET_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    print('Downloading CUB dataset from: {}'.format(BIRDS_DATASET_URL))
    cub_download_location = pathlib.Path('data/CUB_200_2011.tgz')
    urllib.request.urlretrieve(BIRDS_DATASET_URL, cub_download_location)
    # Extract image data
    tar = tarfile.open(cub_download_location, "r:gz")
    if include_text:
        images_save_location = 'data/CUB_200_2011_with_text/images/'
    else:
        images_save_location = 'data/CUB_200_2011'
    tar.extractall(images_save_location)
    tar.close()
    os.remove(cub_download_location)

    if include_text:
        # Download the image captions
        BIRDS_TEXT_GDRIVE_ID = '0B3y_msrWZaXLT1BZdVdycDY5TEE'
        print('Downloading CUB text from Google Drive ID: {}'.format(BIRDS_TEXT_GDRIVE_ID))
        cub_text_download_location = "data/birds.zip"
        gdd.download_file_from_google_drive(file_id=BIRDS_TEXT_GDRIVE_ID,
                                            dest_path=cub_text_download_location,
                                            unzip=True)
        # Move and clean up data
        extracted_text_dir = cub_text_download_location[:-4]
        if os.path.isdir(extracted_text_dir):
            os.rename(extracted_text_dir, 'data/CUB_200_2011_with_text/text/')
        else:
            raise Exception('Expected to find directory {}, but it does not exist'.format(extracted_text_dir))
        os.remove(cub_text_download_location)

def create_tfrecords(tfrecords_dir, image_source_dir, text_source_dir, image_dims, samples_per_shard):
    """ Create the TFRecords dataset
        Arguments:
            tfrecords_dir: str
                Root save location for the TFRecrods
            image_source_dir: str
                Root source directory from which to read the images
            text_source_dir: str
                Root source directory from which to read the text
            image_dims: tuple
                (height (int), width (int))
            samples_per_shard: int
                Number of samples to include in each TFRecord shard
    """
    for subset in ['train', 'test']:
        file_names, class_info, text_embeddings = read_text_subset(subset, text_source_dir)
        file_names = [format_file_name(image_source_dir, file_name) for file_name in file_names]
        text_embeddings = [text_embedding.tobytes() for text_embedding in text_embeddings]
        str_images = get_string_images(file_names)
        shard_iterator = arrange_into_shards(file_names, class_info, text_embeddings, str_images, samples_per_shard)
        write_shards_to_file(shard_iterator, subset, tfrecords_dir)

def format_file_name(image_source_dir, file_name):
    """ Format the file name (to make it compatible with windows) and uses
        utf-8 encoding.
    """
    if os.name == 'nt':
        # Check to see if running in Windows
        file_name = format_for_windows(file_name)
    return os.path.join(image_source_dir, '{}.jpg'.format(file_name)).encode('utf-8')

def read_text_subset(subset, source_dir='data/CUB_200_2011_with_text/text'):
    """ Read the pretrained embedding caption text for the birds and flowers datasets
        as encoded using a pretrained char-CNN-RNN network from:
        https://arxiv.org/abs/1605.05396
    """
    file_names_path = os.path.join(source_dir, subset, 'filenames.pickle')
    file_names = read_pickle(file_names_path)

    class_info_path = os.path.join(source_dir, subset, 'class_info.pickle')
    class_info = read_pickle(class_info_path)

    pretrained_embeddings_path = os.path.join(source_dir, subset, 'char-CNN-RNN-embeddings.pickle')
    char_CNN_RNN_embeddings = read_pickle(pretrained_embeddings_path)

    return file_names, class_info, char_CNN_RNN_embeddings

def read_pickle(path_to_pickle):
    """ Read a pickle file in latin encoding and return the contents """
    with open(path_to_pickle, 'rb') as pickle_file:
        content = pickle.load(pickle_file, encoding='latin1')
    return content

def arrange_into_shards(file_names, class_info_list, text_embeddings, str_images, samples_per_shard):
    """ Convert the listed variables: file_names, class_info_list, text_embeddings, and images
        into chunks to be stored into shards for separate storage as TFRecords.
        Arguments:
            file_names: List
            class_info_list: List
            text_embeddings: List
            str_images: List
            samples_per_shard: int
        Returns:
            iterator
    """
    num_samples = len(file_names)
    num_chunks = math.floor(num_samples / samples_per_shard)
    end_point = num_chunks * num_samples

    assert len(class_info_list) == num_samples and len(text_embeddings) == num_samples and len(file_names) == num_samples, \
        'Expected length of {}'.format(num_samples)

    chunked_file_names = chunk_list(file_names, num_chunks, end_point)
    chunked_class_info_list = chunk_list(class_info_list, num_chunks, end_point)
    chunked_text_embeddings = chunk_list(text_embeddings, num_chunks, end_point)
    chunked_str_images = chunk_list(str_images, num_chunks, end_point)

    return zip(chunked_file_names, chunked_class_info_list, chunked_text_embeddings, chunked_str_images)

def chunk_list(unchuncked_list, num_chunks, end_point):
    """ Split a list up into evenly sized chunks / shards.
        Arguments:
            unchuncked_list: List
                A one-dimensional list
            num_chunks: int
                The number of chunks to create
            end_point: int
                The last index that can be equally chunked
    """
    chunked_list = list(chunks(unchuncked_list[:end_point], num_chunks))
    chunked_list[-1].extend(unchuncked_list[end_point:])
    return chunked_list

def chunks(unchuncked_list, n):
    """ Yield successive n-sized chunks from a list. """
    for i in range(0, len(unchuncked_list), n):
        yield unchuncked_list[i:i + n]

def write_shards_to_file(shard_iterable, subset_name, tfrecords_dir):
    """ Save the TFRecord dataset into separate `shards`.
        Arguments:
            shard_iterable: zip object (iterable)
                Each iteration yields a tuple of 4 objects
            subset_name: str
                Name of the subset (train/test)
            tfrecords_dir: str
                Directory in which the save the TFRecords
    """
    for i, (file_names, classes, text_embeddings, str_images) in enumerate(shard_iterable):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_raw': _bytes_feature(str_images),
                    'names': tf.train.Feature(bytes_list=tf.train.BytesList(value=file_names)),
                    'text': _bytes_feature(text_embeddings),
                    'classes': tf.train.Feature(float_list=tf.train.FloatList(value=classes))
                }
            )
        )
        # Write a separate file to disk for each shard
        mkdir(os.path.join(tfrecords_dir, subset_name))
        record_path_name = os.path.join(tfrecords_dir, subset_name, 'shard-{}.tfrecord'.format(i))
        with tf.io.TFRecordWriter(record_path_name) as writer:
            writer.write(example.SerializeToString())

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def download_flowers():
    """ Download the flowers dataset """
    FLOWERS_DATASET_URL = "www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    print('Downloading the flowers dataset from: {}'.format(FLOWERS_DATASET_URL))

    flowers_download_loc = pathlib.Path('data/flowers/flowers.tgz')
    urllib.request.urlretrieve(FLOWERS_DATASET_URL, flowers_download_loc)
    tar = tarfile.open(flowers_download_loc, "r:gz")
    tar.extractall("data/flowers/images")
    tar.close()
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
        pass

def save_options(options, save_dir):
    """ Save all options to JSON file.
        Arguments:
            options: An object from argparse
            save_dir: String location to save the options
    """
    opt_dict = {}
    for option in vars(options):
        opt_dict[option] = getattr(options, option)

    mkdir(save_dir)
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
    sampled_images = get_images_from_paths(sampled_image_paths, (dataset.height, dataset.width))
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

def get_string_images(image_paths):
    """ Generate a list of string representations of each image """
    str_images_list = []
    for image_path in image_paths:
        image_string = open(image_path, 'rb').read()
        str_images_list.append(image_string)
    return str_images_list

def get_images_from_paths(sampled_image_paths, image_dims):
    """ Given a list of paths to images, and the image dimensions
        to which they belong, load all images into a list.
    """
    height, width = image_dims
    images = []
    for image_path in sampled_image_paths:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=NUM_COLOUR_CHANNELS)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # NOTE: Really not sure why the height and the width need to be
        #       this way around here but above they are reversed
        image = tf.image.resize(image, [height, width])
        images.append(tf.expand_dims(image, axis=0))
    return images

def show_image_list(image_tensor_list, save_dir, name='fake-images.png'):
    """ Visualise and save the tensor image list to file """
    plt.figure(figsize=(10, 10))
    for idx, image_tensor in enumerate(image_tensor_list):
        image_tensor = image_tensor
        x = plt.subplot(5, 5, idx + 1)
        plt.imshow(tf.squeeze(image_tensor, axis=0))
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, name))

def format_for_windows(path_string):
    """ Convert to windows path by replacing `/` with `\` """
    return str(str(path_string).replace('/', '\\'))
