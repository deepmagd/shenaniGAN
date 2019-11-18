import numpy as np
import os
import tensorflow as tf
from utils.utils import get_dataset


def create_dataloaders(args):
    """ """
    dataset = get_dataset(args.dataset_name)

    # # Apparently this is slow - no idea why:
    # # https://www.tensorflow.org/tutorials/load_data/images#load_using_tfdata
    # image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    # train_generator = image_generator.flow_from_directory(
    #     directory=dataset.path,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     target_size=(dataset.height, dataset.width),
    #     classes=dataset.classes
    # )
