import io
import os
import pathlib
import shutil
import tarfile
from typing import Any, List, Dict, Tuple, Optional
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image
from scipy.io import loadmat

from shenanigan.utils.utils import format_file_name, mkdir, normalise, read_pickle

NUM_COLOUR_CHANNELS = 3

IMAGE_SIZE_CONVERSION = {76: 64, 304: 256}


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


def get_record_paths(root_dir: str) -> List[str]:
    """ For a given root directory, find all TFRecords in that structure """
    record_path_names = []
    for root, _, record_names in os.walk(root_dir):
        for record_name in record_names:
            if record_name.endswith('.tfrecord'):
                record_path_name = os.path.join(root, record_name)
                record_path_names.append(record_path_name)
    return record_path_names


def download_dataset(dataset: str):
    if dataset == 'birds-with-text':
        download_cub()
    elif dataset == 'flowers-with-text':
        download_flowers()
    elif dataset == 'xrays':
        """ TODO / NOTE: Since this is a really large download, (and requires the authors' permission)
            we are going to assume that the user will manually download the CheXpert dataset
        """
        raise Exception('Please first download the CheXpert dataset')
    else:
        raise NotImplementedError


def download_cub():
    """ Download the birds dataset (CUB-200-2011) and text """
    BIRDS_DATASET_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"

    cub_download_location = 'data/CUB_200_2011.tgz'
    cub_backup_location = 'data/backup/CUB_200_2011.tgz'

    if os.path.exists(cub_backup_location):
        print('Retrieving CUB dataset from: {}'.format(cub_backup_location))
        shutil.copy(cub_backup_location, cub_download_location)
    else:
        print('Downloading CUB dataset from: {}'.format(BIRDS_DATASET_URL))
        cub_download_location = pathlib.Path('data/CUB_200_2011.tgz')
        urllib.request.urlretrieve(BIRDS_DATASET_URL, cub_download_location)
        mkdir('data/backup')
        shutil.copy(cub_download_location, cub_backup_location)
    tar = tarfile.open(cub_download_location, "r:gz")
    tar.extractall('data/CUB_200_2011_with_text/images/')
    tar.close()
    os.remove(cub_download_location)

    download_captions(
        GDRIVE_ID='0B3y_msrWZaXLT1BZdVdycDY5TEE',
        text_download_location='data/birds.zip',
        backup_location='data/backup/birds.zip',
        res_subdir='CUB_200_2011_with_text',
    )


def download_flowers():
    """ Download the flowers dataset """
    FLOWERS_DATASET_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    print('Downloading the flowers dataset from: {}'.format(FLOWERS_DATASET_URL))

    images_save_location = 'data/flowers_with_text/images/'

    flowers_download_loc = pathlib.Path('data/flowers.tgz')
    urllib.request.urlretrieve(FLOWERS_DATASET_URL, flowers_download_loc)
    tar = tarfile.open(flowers_download_loc, "r:gz")
    tar.extractall(images_save_location)
    tar.close()
    os.remove(flowers_download_loc)

    DATA_SPLITS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    data_splits_download_loc = pathlib.Path(os.path.join(images_save_location, 'setid.mat'))
    urllib.request.urlretrieve(DATA_SPLITS_URL, data_splits_download_loc)

    IMAGE_LABELS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    image_labels_download_loc = pathlib.Path(os.path.join(images_save_location, 'imagelabels.mat'))
    urllib.request.urlretrieve(IMAGE_LABELS_URL, image_labels_download_loc)

    download_captions(
        GDRIVE_ID='0B3y_msrWZaXLaUc0UXpmcnhaVmM',
        text_download_location='data/flowers.zip',
        backup_location='data/backup/flowers.zip',
        res_subdir='flowers_with_text',
    )


def download_captions(GDRIVE_ID: str, text_download_location: str, backup_location: str, res_subdir: str):
    """ The Download and processing for the captions / text part of the dataset """
    extracted_text_dir = text_download_location[:-4]

    if os.path.exists(backup_location):
        print('Retrieving dataset from: {}'.format(backup_location))
        shutil.copy(backup_location, text_download_location)
        with zipfile.ZipFile(backup_location, 'r') as zipfd:
            zipfd.extractall('data/')
    else:
        print('Downloading text from Google Drive ID: {}'.format(GDRIVE_ID))
        gdd.download_file_from_google_drive(file_id=GDRIVE_ID, dest_path=text_download_location, unzip=True)
        mkdir('data/backup')
        shutil.copy(text_download_location, backup_location)

    # Move and clean up data
    if os.path.isdir(extracted_text_dir):
        os.rename(extracted_text_dir, f'data/{res_subdir}/text')
    else:
        raise Exception('Expected to find directory {}, but it does not exist'.format(extracted_text_dir))
    os.remove(text_download_location)


def check_for_xrays(directory: str):
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


def create_image_caption_tfrecords(
    tfrecords_dir: str,
    image_source_dir: str,
    text_source_dir: str,
    bounding_boxes_path: str,
    image_dims_large: Tuple[int, int],
    image_dims_small: Tuple[int, int],
):
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
        file_names, labels, text_embeddings = read_text_subset(subset, text_source_dir)
        file_names = [format_file_name(image_source_dir, file_name) for file_name in file_names]
        bb_map = extract_image_bounding_boxes(image_filenames=file_names, base_path=bounding_boxes_path)
        # Convert to bytes
        text_embeddings = [text_embedding.tobytes() for text_embedding in text_embeddings]
        # NOTE: Ideally we will have a default bb_map returned, but for now we have to skip
        if bb_map is None:
            images_large, images_small = get_byte_images(
                image_paths=file_names, large_image_dims=image_dims_large, small_image_dims=image_dims_small
            )
        else:
            images_large, images_small = get_byte_images(
                image_paths=file_names,
                large_image_dims=image_dims_large,
                small_image_dims=image_dims_small,
                bounding_boxes=bb_map,
                preprocessing='crop',
            )
        wrong_images_large, wrong_images_small = get_wrong_images(
            large_images=images_large, small_images=images_small, labels=labels
        )
        # Arrange and write to file
        shard_iterator = zip(
            *[file_names, images_small, images_large, wrong_images_small, wrong_images_large, text_embeddings, labels]
        )
        write_records_to_file(shard_iterator, subset, tfrecords_dir)


def extract_image_bounding_boxes(image_filenames: list, base_path: str) -> Optional[Dict]:
    """ Returns a map of filename to bounding box in format [x-top, y-top, w, h]
        If no path is provided (None), then simply return None.
    """
    if base_path is None:
        # TODO: Return some default bounding box setting
        return None
    else:
        return extract_bounding_boxes_from_file(image_filenames, base_path)


def extract_bounding_boxes_from_file(image_filenames: str, base_path: str) -> Dict[str, Tuple]:
    bb_df = pd.read_csv(
        os.path.join(base_path, 'bounding_boxes.txt'), names=['idx', 'x', 'y', 'w', 'h'], sep=" "
    ).astype(int)
    imgs_df = pd.read_csv(os.path.join(base_path, 'images.txt'), names=['idx', 'filename'], sep=" ")
    combined_df = imgs_df.merge(bb_df, how='left', on='idx')
    bb_map = {}
    for idx, fn in enumerate(image_filenames):
        bb_map[fn] = (
            combined_df[combined_df['filename'] == "/".join(fn.decode('utf-8').split('/')[-2:])]
            .iloc[:, 2:]
            .values.squeeze()
            .astype(int)
            .tolist()
        )
    return bb_map


def get_wrong_images(large_images: list, small_images: list, labels: list) -> tuple:
    """
    """
    labels = np.array(labels)
    large_images = np.array(large_images)
    wrong_idxs = np.array(list(range(0, len(labels))))
    np.random.shuffle(wrong_idxs)

    error_counter = 0
    while sum(labels == labels[wrong_idxs]) > 0:
        if error_counter == 100:
            raise Exception("Too many iterations in producing 'wrong' images, assuming will not converge")
        collisions = labels == labels[wrong_idxs]
        if sum(collisions) == 1:  # can allow for one duplicate
            wrong_idxs[collisions] = np.random.randint(0, len(labels))
        else:
            wrong_idxs[collisions] = np.random.choice(wrong_idxs[collisions], sum(collisions), replace=False)
        error_counter += 1

    small_images = np.array(small_images)
    return large_images[wrong_idxs].tolist(), small_images[wrong_idxs].tolist()


def create_image_tabular_tfrecords(tfrecords_dir: str, image_source_dir: str, text_source_dir: str, image_dims: tuple):
    """ Create the TFRecords dataset for image-tabular pairs """
    for subset in ['train', 'valid']:
        image_prefix = f'CheXpert-v1.0-small/{subset}/'
        # Tabular encoding
        print('Creating tabular encoding')
        tabular_df = load_tabular_data(os.path.join(image_source_dir, f'{subset}.csv'))
        encoded_tabular_df = encode_tabular_data(tabular_df, image_prefix)
        encoded_tabular_data, image_paths = extract_tabular_as_bytes_lists(
            encoded_tabular_df, prefix=os.path.join('data', 'CheXpert-v1.0-small', 'raw', subset)
        )
        # Convert to bytes
        byte_images = get_byte_images(image_paths=image_paths, image_dims=image_dims)
        # Arrange and write to file
        print('Writing to TFRecords')
        dummy_list = [0] * len(image_paths)
        shard_iterator = zip(*[image_paths, dummy_list, encoded_tabular_data, byte_images])
        write_records_to_file(shard_iterator, subset, tfrecords_dir)
        print('Complete')


def get_byte_images(
    image_paths: List[str],
    large_image_dims: Tuple[int, int],
    small_image_dims: Tuple[int, int],
    preprocessing: str = 'pad',
    **kwargs,
) -> Tuple[List[bytes], List[bytes]]:
    """ Generate a list of byte representations of each image

        if preprocessing == 'crop'
            Required: Dict[string, list] - bounding_boxes
    """
    bounding_boxes = kwargs.get('bounding_boxes')
    if bounding_boxes is None and preprocessing == 'crop':
        raise Exception("bounding boxes required for preprocessing type 'crop'")

    large_image_list = []
    small_image_list = []
    for image_path in image_paths:
        new_img = get_image(image_path, large_image_dims, bounding_boxes, preprocessing)
        byte_image = image_to_bytes(new_img)
        large_image_list.append(byte_image)

        new_img.thumbnail(small_image_dims, Image.ANTIALIAS)
        downsampled_byte_image = image_to_bytes(new_img)
        small_image_list.append(downsampled_byte_image)

    return large_image_list, small_image_list


def image_to_bytes(img: Image) -> bytes:
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    byte_image = img_buffer.getvalue()
    return byte_image


def get_image(image_path: str, image_dims: tuple, bounding_boxes: dict, preprocessing: str = 'pad') -> Image:
    image = Image.open(image_path, 'r')
    if len(image.size) == 2:
        image = image.convert("RGB")
    if preprocessing == 'pad':
        old_size = image.size[:2]
        ratio = max(image_dims) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        image = image.resize(new_size, Image.BICUBIC)
        new_img = Image.new('RGB', image_dims)
        new_img.paste(image, ((image_dims[0] - new_size[0]) // 2, (image_dims[1] - new_size[1]) // 2))
    elif preprocessing == 'crop':
        img = np.array(image)
        bb = bounding_boxes[image_path]
        cx = int(bb[0] + bb[2] / 2)
        cy = int(bb[1] + bb[3] / 2)
        crop_size = int(max(bb[2], bb[3]) * 0.75)
        y1 = max(0, cy - crop_size)
        y2 = min(img.shape[0], cy + crop_size)
        x1 = max(0, cx - crop_size)
        x2 = min(img.shape[1], cx + crop_size)
        img = img[y1:y2, x1:x2, :]
        new_img = np.array(Image.fromarray(img).resize(image_dims, Image.BICUBIC)).astype('uint8')
        new_img = Image.fromarray(new_img)
    else:
        raise Exception(f"No method available for preprpcessing flag '{preprocessing}' when loading byte images")
    return new_img


def read_text_subset(
    subset: str, source_dir: str = 'data/CUB_200_2011_with_text/text'
) -> Tuple[List[str], List[int], List]:
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


def write_records_to_file(example_iterable, subset_name: str, tfrecords_dir: str):
    """ Save the TFRecord dataset with each example in its own TFRecord file.
        Arguments:
            example_iterable: zip object (iterable)
                Each iteration yields a tuple of 4 objects
            subset_name: str
                Name of the subset (train/test)
            tfrecords_dir: str
                Directory in which the save the TFRecords
    """
    for (
        i,
        (file_name, image_small, image_large, wrong_image_small, wrong_image_large, text_embedding, label),
    ) in enumerate(example_iterable):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_small': _bytes_feature(image_small),
                    'image_large': _bytes_feature(image_large),
                    'wrong_image_small': _bytes_feature(wrong_image_small),
                    'wrong_image_large': _bytes_feature(wrong_image_large),
                    'name': _bytes_feature(file_name),
                    'text': _bytes_feature(text_embedding),
                    'label': _int64_feature(label),
                }
            )
        )

        # Write a separate file to disk for each example
        mkdir(os.path.join(tfrecords_dir, subset_name))
        record_path_name = os.path.join(tfrecords_dir, subset_name, 'example-{}.tfrecord'.format(i))
        with tf.io.TFRecordWriter(record_path_name) as writer:
            serialised_example = example.SerializeToString()
            writer.write(serialised_example)


def load_tabular_data(tabular_xray_path: str) -> pd.DataFrame:
    """ Load tabular data and fill all NaN's with a string nan """
    tab_xray_df = pd.read_csv(tabular_xray_path).fillna('nan')
    return tab_xray_df


def build_encoding_map(column: pd.DataFrame) -> Dict[Any, int]:
    """ Build a dictionary which maps each unique item to a categorical integer
        Arguments:
            A column from a Pandas dataframe containing the information
            relating to a single input feature.
    """
    encoding_map = {}
    unique_value_list = column.unique().tolist()
    for idx, unique_value in enumerate(unique_value_list):
        encoding_map[unique_value] = idx
    return encoding_map


def encode_tabular_data(tab_xray_df: pd.DataFrame, image_path_prefix: str) -> pd.DataFrame:
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


def remove_prefix(name: str, prefix: str):
    return name[len(prefix) :]


def extract_tabular_as_bytes_lists(encoded_tabular_df: pd.DataFrame, prefix: str) -> Tuple[List[bytes], List[str]]:
    """ Extract the image paths and feature data from the DataFrame.
        These are converted to bytes and returned as separate lists.
    """
    image_paths = encoded_tabular_df['Path'].values
    image_paths = [os.path.join(prefix, image_path).encode('utf-8') for image_path in image_paths]
    encoded_tabular_lists = encoded_tabular_df.loc[:, encoded_tabular_df.columns != 'Path'].values
    encoded_tabular_lists = [sample.tobytes() for sample in encoded_tabular_lists]
    return encoded_tabular_lists, image_paths


def extract_flowers_labels(path: str) -> List[int]:
    return loadmat(path)['labels'][0, :].tolist()


def extract_flowers_data_split(path: str) -> Tuple[List[int], List[int], List[int]]:
    data = loadmat(path)
    train_ids = data['trnid'][0, :].tolist()
    val_ids = data['valid'][0, :].tolist()
    test_ids = data['tstid'][0, :].tolist()
    return train_ids, val_ids, test_ids


def transform_image(img: tf.Tensor) -> tf.Tensor:
    """ Apply a sequence of tranforms to an image.
        Currently just normalisation.
    """
    img = tf.image.random_flip_left_right(img)

    if img.shape[0] not in IMAGE_SIZE_CONVERSION:
        raise RuntimeError(f'Unsupported image size of {img.shape[0]}')

    if len(img.shape) == 2:
        img = tf.image.random_crop(
            value=img, size=(IMAGE_SIZE_CONVERSION[img.shape[0]], IMAGE_SIZE_CONVERSION[img.shape[1]])
        )
    elif len(img.shape) == 3:
        img = tf.image.random_crop(
            value=img, size=(IMAGE_SIZE_CONVERSION[img.shape[0]], IMAGE_SIZE_CONVERSION[img.shape[1]], img.shape[2])
        )
    else:
        raise RuntimeError(f'Unsupported number of image channels {img.shape}')
    return img * (2.0 / 255) - 1.0


def extract_image_with_text(
    sample: Dict[str, tf.Tensor], index: int, embedding_size: int, num_embeddings_to_sample: int, img_size: str
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """ Return the image, wrong image, and sampled text embeddings """
    image, wrong_image = extract_images(sample=sample, index=index, size=img_size)
    txt = tf.reshape(sample['text'][index], (-1, embedding_size))
    emb_idxs = np.random.choice(txt.shape[0], size=num_embeddings_to_sample, replace=False)
    sampled_txt_embeddings = tf.math.reduce_mean(tf.gather(txt, emb_idxs), axis=0)
    return image, wrong_image, sampled_txt_embeddings


def extract_images(sample: Dict[str, tf.Tensor], index: int, size: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """ NOTE: If include_wrong == False, then we return `None` in place """
    if size in ['small', 'large']:
        image = sample[f'image_{size}'][index]
        wrong_image = sample[f'wrong_image_{size}'][index]
        return image, wrong_image
    else:
        raise Exception(f'There are only two sizes: small and large. Received: {size}')


def tensors_from_sample(
    sample: Dict[str, tf.Tensor],
    batch_size: int,
    text_embedding_size: int,
    num_samples: int,
    augment: bool,
    img_size: str,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """ Extract and format the input samples such that they are ready to be fed into the model """
    image_tensor = []
    wrong_image_tensor = []
    text_tensor = []
    for i in range(batch_size):
        image, wrong_image, text = extract_image_with_text(
            sample=sample,
            index=i,
            embedding_size=text_embedding_size,
            num_embeddings_to_sample=num_samples,
            img_size=img_size,
        )
        image = tf.cast(image, dtype=tf.float32)
        wrong_image = tf.cast(wrong_image, dtype=tf.float32)
        text = tf.cast(text, dtype=tf.float32)

        if augment:
            image = transform_image(image)
            wrong_image = transform_image(wrong_image)

        image_tensor.append(image)
        wrong_image_tensor.append(wrong_image)
        text_tensor.append(text)

    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    wrong_image_tensor = tf.convert_to_tensor(wrong_image_tensor, dtype=tf.float32)
    text_tensor = tf.convert_to_tensor(text_tensor, dtype=tf.float32)

    assert (
        image_tensor.shape == wrong_image_tensor.shape
    ), 'Small real ({}) and wrong ({}) images must have the same dimensions'.format(
        image_tensor.shape, wrong_image_tensor.shape
    )
    return image_tensor, wrong_image_tensor, text_tensor
