import tensorflow as tf

from shenanigan.metrics.inception.model import build
from shenanigan.utils.data_helpers import get_record_paths


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
    img = tf.io.decode_image(parsed_features['image_large'], dtype=tf.float32)
    label = tf.one_hot(parsed_features['label'], depth=classes)
    return img, label

def load_dataset(input_path, batch_size, shuffle_buffer, classes):
    dataset = tf.data.TFRecordDataset(input_path)
    dataset = dataset.map(lambda x: _parse_function(x, classes), num_parallel_calls=16)
    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(1)
    return dataset

def main():
    paths = get_record_paths("../data/CUB_200_2011_with_text/records/train/")
    num_datapoints = len(paths)

    dataset = load_dataset(paths, batch_size=64, shuffle_buffer=10*64, classes=200)

    model = build(classes=200, learning_rate=0.00005, input_shape=(256, 256, 3))

    model.fit(
        dataset,
        epochs=2,
        steps_per_epoch=num_datapoints//64
    )


if __name__ == '__main__':
    main()
