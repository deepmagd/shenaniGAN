import numpy as np
import os
import pathlib
import tensorflow as tf
from utils.data_helpers import download_dataset, check_for_xrays, create_tfrecords, get_record_paths
from utils.data_helpers import NUM_COLOUR_CHANNELS, extract_flowers_labels

DATASETS_DICT = {
    "birds": "BirdsDataset",
    "birds-with-text": "BirdsWithWordsDataset",
    "flowers": "FlowersDataset",
    "flowers-with-text": "FlowersWithWordsDataset",
    "xrays": "XRaysDataset"
}
DATASETS = list(DATASETS_DICT.keys())
AUTOTUNE = tf.data.experimental.AUTOTUNE

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


class StackedGANDataset(object):
    """ Base class for all datasets """
    def __init__(self):
        self.type = None
        self.directory = None
        self.image_label_pairs = None
        self.classes = None
        self.width = None
        self.height = None
        self.num_channels = None

    def get_dims(self):
        return (self.num_channels, self.height, self.width)

    def get_image_label_pairs(self, dataset):
        image_paths = tf.data.Dataset.list_files(str(self.directory/'*/*'))
        if dataset == 'birds':
            self.image_label_pairs = image_paths.map(self.process_path, num_parallel_calls=AUTOTUNE)
        elif dataset == 'flowers':
            labels = extract_flowers_labels(os.path.join(self.directory, 'imagelabels.mat'))
            self.image_label_pairs = [(image_path, label) for image_path, label in zip(image_paths, labels)]
        else:
            raise Exception('Unexpected dataset type: {}'.format(dataset))

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
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        # Resize the image to the desired size
        return tf.image.resize(img, [self.width, self.height])

class BirdsDataset(StackedGANDataset):
    """ Container for the birds dataset properties. """
    def __init__(self):
        super().__init__()
        self.type = 'images'
        self.directory = pathlib.Path(os.path.join('data/CUB_200_2011/CUB_200_2011/images'))
        if not os.path.isdir(self.directory):
            download_dataset(dataset='birds')
        self.classes = np.array(
            [item.name for item in self.directory.glob('*') if os.path.isdir(item.name)]
        )
        self.width = 64
        self.height = 64
        self.num_channels = 3
        self.get_image_label_pairs('birds')


class BirdsWithWordsDataset(StackedGANDataset):
    """ Container for the birds dataset which includes word captions """
    def __init__(self):
        super().__init__()
        # The directory to the TFRecords
        self.type = 'images-with-captions'
        self.width = 64
        self.height = 64
        self.num_channels = 3

        # self.classes = None
        self.feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'wrong_image_raw': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        self.directory = pathlib.Path(
            os.path.join('data/CUB_200_2011_with_text/')
        )
        if not os.path.isdir(self.directory):
            download_dataset(dataset='birds-with-text')
            create_tfrecords(
                dataset_type=self.type,
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images', 'CUB_200_2011', 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                image_dims=(self.height, self.width)
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir

    def parse_dataset(self, subset='train', batch_size=1):
        """ Parse the raw data from the TFRecords and arrange into a readable form
            for the trainer object.
        """
        if subset not in ['train', 'test']:
            raise Exception('Invalid subset type: {}, expected train or test'.format(subset))
        subset_paths = get_record_paths(os.path.join(self.directory, subset))
        subset_obj = tf.data.TFRecordDataset(subset_paths)
        mapped_subset_obj = subset_obj.map(self._parse_example)
        return mapped_subset_obj.batch(batch_size)

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using self.feature_description
        return tf.io.parse_single_example(example_proto, self.feature_description)


class FlowersDataset(StackedGANDataset):
    """ TODO: Container for the birds dataset properties """
    def __init__(self):
        super().__init__()
        self.type = 'images'
        self.directory = pathlib.Path(os.path.join('data/flowers/'))
        if not os.path.isdir(self.directory):
            download_dataset(dataset='flowers')
        self.classes = list(set(extract_flowers_labels(os.path.join(self.directory, 'imagelabels.mat'))))
        self.width = 64  # TODO: Double check that I have this right
        self.height = 64
        self.num_channels = 3
        self.get_image_label_pairs('flowers')

class FlowersWithWordsDataset(StackedGANDataset):
    """ Container for the birds dataset which includes word captions """
    def __init__(self):
        super().__init__()
        # The directory to the TFRecords
        self.type = 'images-with-captions'
        self.width = 64
        self.height = 64
        self.num_channels = 3

        self.feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'wrong_image_raw': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        self.directory = pathlib.Path(
            os.path.join('data/flowers_with_text/')
        )
        if not os.path.isdir(self.directory):
            download_dataset(dataset='flowers-with-text')
            import sys
            sys.exit()
            create_tfrecords(
                dataset_type=self.type,
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images', 'CUB_200_2011', 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                image_dims=(self.height, self.width)
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir

    def parse_dataset(self, subset='train', batch_size=1):
        """ Parse the raw data from the TFRecords and arrange into a readable form
            for the trainer object.
        """
        if subset not in ['train', 'test']:
            raise Exception('Invalid subset type: {}, expected train or test'.format(subset))
        subset_paths = get_record_paths(os.path.join(self.directory, subset))
        subset_obj = tf.data.TFRecordDataset(subset_paths)
        mapped_subset_obj = subset_obj.map(self._parse_example)
        return mapped_subset_obj.batch(batch_size)

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using self.feature_description
        return tf.io.parse_single_example(example_proto, self.feature_description)

class XRaysDataset(StackedGANDataset):
    """ XXX: Container for the x-rays dataset properties """
    def __init__(self):
        super().__init__()
        self.type = 'images-with-tabular'
        # NOTE: width and height are for the small dataset for now
        self.width = 390
        self.height = 320
        self.num_channels = 1

        self.feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

        base_directory = 'data/CheXpert-v1.0-small'
        if not os.path.isdir(os.path.join(base_directory, 'raw')):
            check_for_xrays(directory='data/CheXpert-v1.0-small')

        self.directory = os.path.join(base_directory, 'records')
        if not os.path.isdir(self.directory):
            create_tfrecords(
                dataset_type=self.type,
                tfrecords_dir=self.directory,
                image_source_dir=os.path.join(base_directory, 'raw'),
                text_source_dir=os.path.join(base_directory, 'raw'),
                image_dims=(self.height, self.width)
            )

    def parse_dataset(self, subset='train', batch_size=1):
        """ Parse the raw data from the TFRecords and arrange into a readable form
            for the trainer object.
        """
        if subset not in ['train', 'valid']:
            raise Exception('Invalid subset type: {}, expected train or valid'.format(subset))

        subset_paths = get_record_paths(os.path.join(self.directory, subset))
        subset_obj = tf.data.TFRecordDataset(subset_paths)
        mapped_subset_obj = subset_obj.map(self._parse_example)
        return mapped_subset_obj.batch(batch_size)

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using self.feature_description
        return tf.io.parse_single_example(example_proto, self.feature_description)
