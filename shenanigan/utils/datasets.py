import os
import pathlib

import numpy as np
import tensorflow as tf

from shenanigan.utils.data_helpers import (NUM_COLOUR_CHANNELS,
                                           check_for_xrays,
                                           create_image_caption_tfrecords,
                                           create_image_tabular_tfrecords,
                                           download_dataset,
                                           extract_flowers_labels,
                                           get_record_paths)

DATASETS_DICT = {
    "birds-with-text": "BirdsWithWordsDataset",
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


class StackGANDataset(object):
    """ Base class for all datasets """
    def __init__(self):
        self.type = None
        self.directory = None
        self.image_dims_small = (None, None)
        self.image_dims_large = (None, None)
        self.num_channels = None
        self.text_embedding_dim = None

        self.feature_description = {
            'image_small': tf.io.FixedLenFeature([], tf.string),
            'image_large': tf.io.FixedLenFeature([], tf.string),
            'wrong_image_small': tf.io.FixedLenFeature([], tf.string),
            'wrong_image_large': tf.io.FixedLenFeature([], tf.string),
            'name': tf.io.FixedLenFeature([], tf.string),
            'text': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }

    def get_small_dims(self):
        """ Return in the form (dept, height, width) """
        return (self.num_channels, self.image_dims_small[1], self.image_dims_small[0])

    def get_large_dims(self):
        """ Return in the form (dept, height, width) """
        return (self.num_channels, self.image_dims_large[1], self.image_dims_large[0])

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
        parsed_features = tf.io.parse_single_example(example_proto, self.feature_description)
        parsed_features['image_small'] = tf.io.decode_image(parsed_features['image_small'], dtype=tf.float32) * 255
        parsed_features['image_large'] = tf.io.decode_image(parsed_features['image_large'], dtype=tf.float32) * 255
        parsed_features['wrong_image_small'] = tf.io.decode_image(parsed_features['wrong_image_small'], dtype=tf.float32) * 255
        parsed_features['wrong_image_large'] = tf.io.decode_image(parsed_features['wrong_image_large'], dtype=tf.float32) * 255
        parsed_features['text'] = tf.io.decode_raw(parsed_features['text'], out_type=tf.float32)
        return parsed_features

class BirdsWithWordsDataset(StackGANDataset):
    """ Container for the birds dataset which includes word captions """
    def __init__(self):
        super().__init__()
        self.type = 'images-with-captions'
        self.image_dims_small = (76, 76)
        self.image_dims_large = (304, 304)
        self.num_channels = 3
        self.text_embedding_dim = 1024

        self.directory = pathlib.Path(
            os.path.join('data/CUB_200_2011_with_text/')
        )
        if not os.path.isdir(self.directory):
            download_dataset(dataset='birds-with-text')
            create_image_caption_tfrecords(
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images', 'CUB_200_2011', 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                bounding_boxes_path=os.path.join(self.directory, 'images', 'CUB_200_2011'),
                image_dims_large=self.image_dims_large,
                image_dims_small=self.image_dims_small
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir


class FlowersWithWordsDataset(StackGANDataset):
    """ Container for the birds dataset which includes word captions """
    def __init__(self):
        super().__init__()
        self.type = 'images-with-captions'
        self.image_dims_small = (64, 64)
        self.image_dims_large = (256, 256)
        self.num_channels = 3
        self.text_embedding_dim = 1024

        self.directory = pathlib.Path(
            os.path.join('data/flowers_with_text/')
        )
        if not os.path.isdir(self.directory):
            download_dataset(dataset='flowers-with-text')
            create_image_caption_tfrecords(
                tfrecords_dir=os.path.join(self.directory, 'records'),
                image_source_dir=os.path.join(self.directory, 'images'),
                text_source_dir=os.path.join(self.directory, 'text'),
                bounding_boxes_path=None,
                image_dims_large=self.image_dims_large,
                image_dims_small=self.image_dims_small
            )

        records_dir = os.path.join(self.directory, 'records')
        if os.path.isdir(records_dir):
            self.directory = records_dir


class XRaysDataset(StackGANDataset):
    """ XXX: Container for the x-rays dataset properties """
    def __init__(self):
        super().__init__()
        # TODO: Rename valid to test in data download
        self.type = 'images-with-tabular'
        # NOTE: width and height are for the small dataset for now
        self.image_dims_small = (None, None)
        self.image_dims_large = (390, 320)
        self.num_channels = 1
        # TODO: Set this
        self.text_embedding_dim = None

        base_directory = 'data/CheXpert-v1.0-small'
        if not os.path.isdir(os.path.join(base_directory, 'raw')):
            check_for_xrays(directory='data/CheXpert-v1.0-small')

        self.directory = os.path.join(base_directory, 'records')
        if not os.path.isdir(self.directory):
            create_image_tabular_tfrecords(
                tfrecords_dir=self.directory,
                image_source_dir=os.path.join(base_directory, 'raw'),
                text_source_dir=os.path.join(base_directory, 'raw'),
                image_dims=(self.height, self.width)
            )
