import numpy as np
import os
import pathlib
import tensorflow as tf
from utils.data_helpers import download_dataset, create_tfrecords, get_record_paths
from utils.data_helpers import NUM_COLOUR_CHANNELS

DATASETS_DICT = {
    "birds": "BirdsDataset",
    "birds-with-text": "BirdsWithWordsDataset",
    "flowers": "FlowersDataset",
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
        self.type = 'images'
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
        self.type = 'images-with-captions'
        self.width = 500
        self.height = 364
        self.num_channels = 3

        # self.classes = None
        self.feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
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
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images', 'CUB_200_2011', 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                image_dims=(self.height, self.width)
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir

    # def __len__(self):
    #     return num_tfrecords_in_dir(os.path.join(self.directory, 'train'))

    def parse_dataset(self, subset='train'):
        """ Parse the raw data from the TFRecords and arrange into a readable form
            for the trainer object.
        """
        if not subset in ['train', 'test']:
            raise Exception('Invalid subset type: {}, expected train or test'.format(subset))

        subset_paths = get_record_paths(os.path.join(self.directory, subset))
        subset_obj = tf.data.TFRecordDataset(subset_paths)
        mapped_subset_obj = subset_obj.map(self._parse_example)
        return mapped_subset_obj

    def _parse_example(self, example_proto):
        # Parse the input tf.Example proto using self.feature_description
        return tf.io.parse_single_example(example_proto, self.feature_description)


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
