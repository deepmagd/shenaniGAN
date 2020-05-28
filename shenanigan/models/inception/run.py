import os

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

from shenanigan.models.inception.model import build
from shenanigan.utils.data_helpers import get_record_paths
from shenanigan.utils.utils import mkdir

def _parse_function(proto, classes):
    f = {
        'image_small': tf.io.FixedLenFeature([], tf.string),
        'image_large': tf.io.FixedLenFeature([], tf.string),
        'wrong_image_small': tf.io.FixedLenFeature([], tf.string),
        'wrong_image_large': tf.io.FixedLenFeature([], tf.string),
        'name': tf.io.FixedLenFeature([], tf.string),
        'text': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, f)
    img = tf.io.decode_image(parsed_features['image_large'], dtype=tf.float32) * 255
    img = preprocess_input(img)
    label = tf.one_hot(parsed_features['label'], depth=classes)
    return img, label

def load_dataset(input_path, batch_size, shuffle_buffer, classes):
    dataset = tf.data.TFRecordDataset(input_path)
    dataset = dataset.map(lambda x: _parse_function(x, classes))
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(5)
    return dataset

def _get_record_paths(root_path):
    paths = get_record_paths(root_path)
    return paths, len(paths)

def run(experiment_name, dataset_name, settings):

    if dataset_name == 'birds-with-text':
        train_paths, num_train_samples = _get_record_paths("data/CUB_200_2011_with_text/records/train/")
        test_paths, num_test_samples = _get_record_paths("data/CUB_200_2011_with_text/records/test/")
    else:
        raise Exception(f"Unsupported dataset name of type '{dataset_name}'")

    train_loader = load_dataset(
        train_paths,
        batch_size=settings[dataset_name]['batch_size'],
        shuffle_buffer=settings[dataset_name]['buffer_size'],
        classes=settings[dataset_name]['num_classes']
    )
    valid_loader = load_dataset(
        test_paths,
        batch_size=settings[dataset_name]['batch_size'],
        shuffle_buffer=settings[dataset_name]['buffer_size'],
        classes=settings[dataset_name]['num_classes']
    )

    model = build(
        classes=settings[dataset_name]['num_classes'],
        learning_rate=settings[dataset_name]['learning_rate'],
        input_shape=(
            settings[dataset_name]['image_shape']['H'],
            settings[dataset_name]['image_shape']['W'],
            settings[dataset_name]['image_shape']['C']
        )
    )

    model.fit(
        train_loader,
        epochs=settings[dataset_name]['epochs'],
        steps_per_epoch=num_train_samples//settings[dataset_name]['batch_size'],
        validation_data=valid_loader,
        validation_steps=num_test_samples//settings[dataset_name]['batch_size']
    )

    save_path = f"results/{experiment_name}/inception"
    mkdir(save_path)
    model.save(os.path.join(save_path, "model"))
