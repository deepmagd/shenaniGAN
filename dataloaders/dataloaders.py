import numpy as np
import os
import tensorflow as tf
from utils.datasets import get_dataset


def create_dataloaders(args):
    """ Create traing and validation set generators """
    dataset = get_dataset(args.dataset_name)

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
        batch_size=args.batch_size,
        shuffle=True,
        target_size=(dataset.height, dataset.width),
        classes=dataset.classes,
        subset='training'
    )
    val_generator = image_generator.flow_from_directory(
        directory=dataset.directory,
        batch_size=args.batch_size,
        shuffle=True,
        target_size=(dataset.height, dataset.width),
        classes=dataset.classes,
        subset='validation'
    )
    return train_generator, val_generator, dataset.get_dims()
