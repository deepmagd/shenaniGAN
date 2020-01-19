import glob
from google_drive_downloader import GoogleDriveDownloader as gdd
import math
from matplotlib import pyplot as plt
import os
import pathlib
from random import randint
import tarfile
import tensorflow as tf
import urllib.request
from utils.utils import chunk_list, format_file_name, read_pickle, mkdir

NUM_COLOUR_CHANNELS = 3


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

def get_record_paths(root_dir):
    """ For a given root directory, find all TFRecords in that structure """
    record_path_names = []
    for root, _, record_names in os.walk(root_dir):
        for record_name in record_names:
            if record_name.endswith('.tfrecord'):
                record_path_name = os.path.join(root, record_name)
                record_path_names.append(record_path_name)
    return record_path_names

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
    for root, _, names in os.walk(data_dir):
        image_paths = [os.path.join(root, name) for name in names]

    sampled_image_paths = []
    for _ in range(num_paths):
        sample_idx = randint(0, len(image_paths) - 1)
        sampled_image_paths.append(image_paths[sample_idx])
    return sampled_image_paths

def show_image_list(image_tensor_list, save_dir, name='fake-images.png'):
    """ Visualise and save the tensor image list to file """
    plt.figure(figsize=(10, 10))
    for idx, image_tensor in enumerate(image_tensor_list):
        image_tensor = image_tensor
        x = plt.subplot(5, 5, idx + 1)
        plt.imshow(tf.squeeze(image_tensor, axis=0))
        plt.axis('off')
    plt.savefig(os.path.join(save_dir, name))

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
