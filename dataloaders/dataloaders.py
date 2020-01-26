import numpy as np
import os
import tensorflow as tf
from utils.datasets import get_dataset
from utils.utils import num_tfrecords_in_dir


def create_dataloaders(args):
    """ Create traing and validation set generators """
    dataset = get_dataset(args.dataset_name)
    if dataset.type == 'images':
        return image_loaders(dataset, args.batch_size)
    elif dataset.type == 'images-with-captions':
        print("HEREH HERH HERE")
        return image_with_captions_loaders(dataset)
    else:
        raise Exception('Unexpected dataset type: {}'.format(dataset.type))

def image_with_captions_loaders(dataset):
    """ Read, prepare, and present the TFRecord data """
    train_loader = ImageTextDataLoader(
        dataset_object=dataset,
        subset='train'
    )
    test_loader = ImageTextDataLoader(
        dataset_object=dataset,
        subset='test'
    )
    return train_loader, test_loader, dataset.get_dims()

def image_loaders(dataset, batch_size):
    # No data augmentation for now
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        validation_split=0.1
    )
    train_generator = image_generator.flow_from_directory(
        directory=dataset.directory,
        batch_size=batch_size,
        shuffle=True,
        target_size=(dataset.height, dataset.width),
        classes=dataset.classes,
        subset='training'
    )
    val_generator = image_generator.flow_from_directory(
        directory=dataset.directory,
        batch_size=batch_size,
        shuffle=True,
        target_size=(dataset.height, dataset.width),
        classes=dataset.classes,
        subset='validation'
    )
    return train_generator, val_generator, dataset.get_dims()


class ImageTextDataLoader(object):
    """ Define a dataloader for image-text pairs """
    def __init__(self, dataset_object, subset):
        """ Initialise a ImageTextDataLoader object for either the train
            or test subsets using a BirdsWithWordsDataset object.
        """
        # self.index = 0
        self.subset = subset
        self.dataset_object = dataset_object
        self.parsed_subset = self.dataset_object.parse_dataset(self.subset)

    def __call__(self):
        for sample in self.parsed_subset:
            yield sample

    def __len__(self):
        return num_tfrecords_in_dir(os.path.join(self.dataset_object.directory, self.subset))
