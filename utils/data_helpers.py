import io
import os
import pathlib
import shutil
import tarfile
import urllib.request
from random import randint

import pandas as pd
import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from matplotlib import pyplot as plt
from PIL import Image

from utils.utils import format_file_name, mkdir, normalise, read_pickle

NUM_COLOUR_CHANNELS = 3


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_tfrecords(dataset_type, tfrecords_dir, image_source_dir, text_source_dir, image_dims):
    """ Create the TFRecords dataset
        Arguments:
            dataset_type: str
                The dataset type which dictates how we create the records
            tfrecords_dir: str
                Root save location for the TFRecrods
            image_source_dir: str
                Root source directory from which to read the images
            text_source_dir: str
                Root source directory from which to read the text
            image_dims: tuple
                (height (int), width (int))
    """
    if dataset_type == 'images-with-captions':
        create_image_caption_tfrecords(tfrecords_dir, image_source_dir, text_source_dir, image_dims)
    elif dataset_type == 'images-with-tabular':
        create_image_tabular_tfrecords(tfrecords_dir, image_source_dir, text_source_dir, image_dims)
    else:
        raise Exception(f'{dataset_type} is not a recognised dataset type')

def create_image_caption_tfrecords(tfrecords_dir, image_source_dir, text_source_dir, image_dims):
    """ Create the TFRecords dataset for image-caption pairs
        Arguments:
            tfrecords_dir: str
                Root save location for the TFRecrods
            image_source_dir: str
                Root source directory from which to read the images
            text_source_dir: str
                Root source directory from which to read the text
            image_dims: tuple
                (height (int), width (int))
    """
    for subset in ['train', 'test']:
        # Read from file and format
        file_names, class_info, text_embeddings = read_text_subset(subset, text_source_dir)
        file_names = [format_file_name(image_source_dir, file_name) for file_name in file_names]
        # Convert to bytes
        text_embeddings = [text_embedding.tobytes() for text_embedding in text_embeddings]
        byte_images = get_byte_images(image_paths=file_names, image_dims=image_dims)
        # Arrange and write to file
        shard_iterator = zip(*[file_names, class_info, text_embeddings, byte_images])
        write_records_to_file(shard_iterator, subset, tfrecords_dir)

def create_image_tabular_tfrecords(tfrecords_dir, image_source_dir, text_source_dir, image_dims):
    """ Create the TFRecords dataset for image-tabular pairs """
    for subset in ['train', 'valid']:
        image_prefix = f'CheXpert-v1.0-small/{subset}/'
        # Tabular encoding
        print('Creating tabular encoding')
        tabular_df = load_tabular_data(os.path.join(image_source_dir, f'{subset}.csv'))
        encoded_tabular_df = encode_tabular_data(tabular_df, image_prefix)
        encoded_tabular_data, image_paths = extract_tabular_as_bytes_lists(
            encoded_tabular_df,
            prefix=os.path.join('data', 'CheXpert-v1.0-small', 'raw', subset)
        )
        # Convert to bytes
        byte_images = get_byte_images(image_paths=image_paths, image_dims=image_dims)
        # Arrange and write to file
        print('Writing to TFRecords')
        dummy_list = [0] * len(image_paths)
        shard_iterator = zip(*[image_paths, dummy_list, encoded_tabular_data, byte_images])
        write_records_to_file(shard_iterator, subset, tfrecords_dir)
        print('Complete')

def download_dataset(dataset):
    if dataset == 'birds':
        download_cub()
    elif dataset == 'birds-with-text':
        download_cub(include_text=True)
    elif dataset == 'flowers':
        download_flowers()
    elif dataset == 'xrays':
        """ TODO / NOTE: Since this is a really large download, (and requires the authors' permission)
            we are going to assume that the user will manually download the CheXpert dataset
        """
        raise Exception('Please first download the CheXpert dataset')
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

def check_for_xrays(directory):
    """ Check to see if the xray dataset has been downloaded at all.
        Raise an exception if it hasn't. If it has, move it to raw.
    """
    train_location = os.path.join(directory, 'train')
    valid_location = os.path.join(directory, 'valid')
    raw_location = os.path.join(directory, 'raw')

    if not os.path.isdir(train_location) or not os.path.isdir(valid_location):
        raise Exception('Please first download the CheXpert dataset')

    mkdir(raw_location)
    shutil.move(train_location, raw_location)
    shutil.move(valid_location, raw_location)
    shutil.move(f'{train_location}.csv', raw_location)
    shutil.move(f'{valid_location}.csv', raw_location)

def get_byte_images(image_paths, image_dims):
    """ Generate a list of byte representations of each image """
    byte_images_list = []
    for image_path in image_paths:
        image = Image.open(image_path, 'r')
        old_size = image.size[:2]
        ratio = max(image_dims)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        image = image.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new('RGB', image_dims)
        new_img.paste(image, ((image_dims[0]-new_size[0])//2,
                              (image_dims[1]-new_size[1])//2))
        img_buffer = io.BytesIO()
        new_img.save(img_buffer, format='PNG')
        byte_image = img_buffer.getvalue()
        byte_images_list.append(byte_image)
    return byte_images_list

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

def sample_real_images(num_images, dataset_object):
    """ Randomly sample images (with replacement) from all
        available images in the data directory.
        Arguments:
            num_images: int
                The number of images to sample from the dataset
            dataset_object:
                The dataset class
    """
    sampled_image_paths = sample_image_paths(dataset_object.directory, num_images)
    sampled_images = get_images_from_paths(sampled_image_paths, (dataset_object.height, dataset_object.width))
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

def write_records_to_file(example_iterable, subset_name, tfrecords_dir):
    """ Save the TFRecord dataset with each example in its own TFRecord file.
        Arguments:
            example_iterable: zip object (iterable)
                Each iteration yields a tuple of 4 objects
            subset_name: str
                Name of the subset (train/test)
            tfrecords_dir: str
                Directory in which the save the TFRecords
    """
    for i, (file_name, label, text_embedding, str_image) in enumerate(example_iterable):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_raw': _bytes_feature(str_image),
                    'name': _bytes_feature(file_name),
                    'text': _bytes_feature(text_embedding),
                    'label': _int64_feature(label)
                }
            )
        )

        # Write a separate file to disk for each example
        mkdir(os.path.join(tfrecords_dir, subset_name))
        record_path_name = os.path.join(tfrecords_dir, subset_name, 'example-{}.tfrecord'.format(i))
        with tf.io.TFRecordWriter(record_path_name) as writer:
            serialised_example = example.SerializeToString()
            writer.write(serialised_example)

def load_tabular_data(tabular_xray_path):
    """ Load tabular data and fill all NaN's with a string nan """
    tab_xray_df = pd.read_csv(tabular_xray_path).fillna('nan')
    return tab_xray_df

def build_encoding_map(column):
    """ Build a dictionary which maps each unique item to a categorical integer
        Arguments:
            column: pd.DataFrame
                A column from a Pandas dataframe containing the information
                relating to a single input feature.
    """
    encoding_map = {}
    unique_value_list = column.unique().tolist()
    for idx, unique_value in enumerate(unique_value_list):
        encoding_map[unique_value] = idx
    return encoding_map

def encode_tabular_data(tab_xray_df, image_path_prefix:str):
    """ Encode the tabular data so that it is represented in one-hot
        encoding for categorical variables, and normalised for continious
        variables
    """
    ignores = ['Path']
    normalises = ['Age']
    encoded_df = pd.DataFrame({'Path': tab_xray_df['Path'].values})

    for column in tab_xray_df:
        if column not in ignores and column not in normalises:
            # One-hot encode the categorical data
            one_hot_subset_df = pd.get_dummies(tab_xray_df[column], prefix=column)
            encoded_df = encoded_df.join(one_hot_subset_df)
        if column in normalises and column not in ignores:
            # Continious variable, simply normalise
            norm_values = normalise(tab_xray_df[column].values)
            encoded_df = encoded_df.join(pd.DataFrame({column: norm_values}))
    encoded_df['Path'] = encoded_df['Path'].apply(lambda x: remove_prefix(x, prefix=image_path_prefix))
    return encoded_df

def remove_prefix(name, prefix):
    return name[len(prefix):]

def extract_tabular_as_bytes_lists(encoded_tabular_df, prefix):
    """ Extract the image paths and feature data from the DataFrame.
        These are converted to bytes and returned as separate lists.
    """
    image_paths = encoded_tabular_df['Path'].values
    image_paths = [os.path.join(prefix, image_path).encode('utf-8') for image_path in image_paths]
    encoded_tabular_lists = encoded_tabular_df.loc[:, encoded_tabular_df.columns != 'Path'].values
    encoded_tabular_lists = [sample.tobytes() for sample in encoded_tabular_lists]
    return encoded_tabular_lists, image_paths
