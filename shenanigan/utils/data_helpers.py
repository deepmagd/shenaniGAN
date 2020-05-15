import io
import os
import pathlib
import shutil
import tarfile
from tqdm import tqdm
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image
from scipy.io import loadmat

from shenanigan.utils.utils import (format_file_name, mkdir, normalise,
                                    read_pickle)

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

def get_record_paths(root_dir):
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
        res_subdir='CUB_200_2011_with_text'
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
        res_subdir='flowers_with_text'
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
        gdd.download_file_from_google_drive(file_id=GDRIVE_ID,
                                            dest_path=text_download_location,
                                            unzip=True)
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
        Raise an exception if it hasn't. If it has, move it to raw and rename valid to test.
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

    mkdir(os.path.join(raw_location, 'test'))
    shutil.move(
        src=os.path.join(raw_location, 'valid.csv'),
        dst=os.path.join(raw_location, 'test.csv')
    )
    shutil.move(
        src=os.path.join(raw_location, 'valid'),
        dst=os.path.join(raw_location, 'test')
    )

def create_image_caption_tfrecords(tfrecords_dir: str, image_source_dir: str, text_source_dir: str,
                                   bounding_boxes_path: str, image_dims_large: tuple, image_dims_small: tuple):
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
                image_paths=file_names,
                large_image_dims=image_dims_large,
                small_image_dims=image_dims_small
            )
        else:
            images_large, images_small = get_byte_images(
                image_paths=file_names,
                large_image_dims=image_dims_large,
                small_image_dims=image_dims_small,
                bounding_boxes=bb_map,
                preprocessing='crop'
            )
        wrong_images_large, wrong_images_small = get_wrong_images(
            large_images=images_large,
            small_images=images_small,
            labels=labels
        )
        # Arrange and write to file
        shard_iterator = zip(*[file_names, images_small, images_large, wrong_images_small,
                               wrong_images_large, text_embeddings, labels])
        write_records_to_file(shard_iterator, subset, tfrecords_dir)

def extract_image_bounding_boxes(image_filenames: list, base_path: str):
    """ Returns a map of filename to bounding box in format [x-top, y-top, w, h]
        If no path is provided (None), then simply return None.
    """
    if base_path is None:
        # TODO: Return some default bounding box setting
        return None
    else:
        return extract_bounding_boxes_from_file(image_filenames, base_path)

def extract_bounding_boxes_from_file(image_filenames: str, base_path: str) -> dict:
    bb_df = pd.read_csv(
        os.path.join(base_path, 'bounding_boxes.txt'),
        names=['idx', 'x', 'y', 'w', 'h'], sep=" "
    ).astype(int)
    imgs_df = pd.read_csv(os.path.join(base_path, 'images.txt'), names=['idx', 'filename'], sep=" ")
    combined_df = imgs_df.merge(bb_df, how='left', on='idx')
    bb_map = {}
    for idx, fn in enumerate(image_filenames):
        bb_map[fn] = combined_df[
            combined_df['filename'] == "/".join(fn.decode('utf-8').split('/')[-2:])
        ].iloc[:, 2:].values.squeeze().astype(int).tolist()
    return bb_map

def get_wrong_images(large_images: list, small_images: list, labels: list) -> tuple:
    """ Generate two corresponding lists where the image order has been shuffled
        so that they are no longer in the correct order with the text and are not
        in the same label class. As a result we have "wrong" images for the
        corresponding text.
    """
    labels = np.array(labels)
    large_images = np.array(large_images)
    wrong_idxs = np.array(list(range(0, len(labels))))
    np.random.shuffle(wrong_idxs)

    error_counter = 0
    while (sum(labels == labels[wrong_idxs]) > 0):
        if (error_counter == 100):
            raise Exception("Too many iterations in producing 'wrong' images, assuming will not converge")
        collisions = labels == labels[wrong_idxs]
        if (sum(collisions) == 1):  # can allow for one duplicate
            wrong_idxs[collisions] = np.random.randint(0, len(labels))
        else:
            wrong_idxs[collisions] = np.random.choice(wrong_idxs[collisions], sum(collisions), replace=False)
        error_counter += 1

    small_images = np.array(small_images)
    return large_images[wrong_idxs].tolist(), small_images[wrong_idxs].tolist()

def create_image_tabular_tfrecords(tfrecords_dir: str, image_source_dir: str, text_source_dir: str,
                                   bounding_boxes_path: str, image_dims_large: tuple, image_dims_small: tuple):
    """ Create the TFRecords dataset for image-tabular pairs """
    for subset in ['train', 'test']:
        image_prefix = f'CheXpert-v1.0-small/{subset}/'
        # Tabular encoding
        print('Creating tabular encoding ...')
        tabular_df = load_tabular_data(os.path.join(image_source_dir, f'{subset}.csv'))
        encoded_tabular_data, image_paths = extract_tabular_as_bytes_lists(
            encoded_tabular_df=encode_tabular_data(tabular_df, image_prefix),
            prefix=os.path.join('data', 'CheXpert-v1.0-small', 'raw', subset)
        )
        # NOTE: Since we do not have labels, I shall uniquely label all patients
        dummy_labels = range(len(encoded_tabular_data))
        print('Reading and cropping images ...')
        # NOTE: Ideally we will have a default bb_map returned, but for now we have to skip
        bb_map = extract_image_bounding_boxes(image_filenames=image_paths, base_path=bounding_boxes_path)
        if bb_map is None:
            images_large, images_small = get_byte_images(
                image_paths=image_paths,
                large_image_dims=image_dims_large,
                small_image_dims=image_dims_small
            )
        else:
            images_large, images_small = get_byte_images(
                image_paths=image_paths,
                large_image_dims=image_dims_large,
                small_image_dims=image_dims_small,
                bounding_boxes=bb_map,
                preprocessing='crop'
            )
        print("Generating wrong images ...")
        wrong_images_large, wrong_images_small = get_wrong_images(
            large_images=images_large,
            small_images=images_small,
            labels=dummy_labels
        )
        # Arrange and write to file
        print('Writing to TFRecords ,,,')
        shard_iterator = zip(*[image_paths, images_small, images_large, wrong_images_small,
                               wrong_images_large, encoded_tabular_data, dummy_labels])
        write_records_to_file(shard_iterator, subset, tfrecords_dir)
        print(f'Complete for {subset} set')


def get_byte_images(image_paths: list, large_image_dims: tuple,
                    small_image_dims: tuple, preprocessing: str = 'pad', **kwargs):
    """ Generate a list of byte representations of each image

        if preprocessing == 'crop'
            Required: Dict[string, list] - bounding_boxes
    """
    bounding_boxes = kwargs.get('bounding_boxes')
    if bounding_boxes is None and preprocessing == 'crop':
        raise Exception("bounding boxes required for preprocessing type 'crop'")

    large_image_list = []
    small_image_list = []
    for image_path in tqdm(image_paths):
        new_img = get_image(image_path, large_image_dims, bounding_boxes, preprocessing)
        byte_image = image_to_bytes(new_img)
        large_image_list.append(byte_image)

        new_img.thumbnail(small_image_dims, Image.ANTIALIAS)
        downsampled_byte_image = image_to_bytes(new_img)
        small_image_list.append(downsampled_byte_image)

    return large_image_list, small_image_list

def image_to_bytes(img):
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    byte_image = img_buffer.getvalue()
    return byte_image

def get_image(image_path: str, image_dims: tuple, bounding_boxes: dict, preprocessing: str = 'pad'):
    image = Image.open(image_path, 'r')
    if len(image.size) == 2:
        image = image.convert("RGB")
    if preprocessing == 'pad':
        old_size = image.size[:2]
        ratio = max(image_dims)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        image = image.resize(new_size, Image.BICUBIC)
        new_img = Image.new('RGB', image_dims)
        new_img.paste(image, ((image_dims[0]-new_size[0])//2,
                              (image_dims[1]-new_size[1])//2))
    elif preprocessing == 'crop':
        img = np.array(image)
        bb = bounding_boxes[image_path]
        cx = int(bb[0]+bb[2]/2)
        cy = int(bb[1]+bb[3]/2)
        crop_size = int(max(bb[2], bb[3])*0.75)
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
    for i, (file_name, image_small, image_large, wrong_image_small,
            wrong_image_large, text_embedding, label) in enumerate(example_iterable):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image_small': _bytes_feature(image_small),
                    'image_large': _bytes_feature(image_large),
                    'wrong_image_small': _bytes_feature(wrong_image_small),
                    'wrong_image_large': _bytes_feature(wrong_image_large),
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

def encode_tabular_data(tab_xray_df, image_path_prefix):
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

def extract_flowers_labels(path: str):
    return loadmat(path)['labels'][0, :].tolist()

def extract_flowers_data_split(path: str):
    data = loadmat(path)
    train_ids = data['trnid'][0, :].tolist()
    val_ids = data['valid'][0, :].tolist()
    test_ids = data['tstid'][0, :].tolist()
    return train_ids, val_ids, test_ids

def transform_image(img):
    """ Apply a sequence of tranforms to an image.
        Currently just normalisation.
    """
    return img * (2./255) - 1.

def extract_image_with_text(sample: dict, index: int, embedding_size: int,
                            num_embeddings_to_sample: int):
    """ Return a list of the extracted field from the sample. These tend to take the form:
        (image_size1, wrong_image_size1, image_size2, wrong_image_size2, ...., text)
        where wrong images could be `None`
    """
    extracted_fields = []
    for size in ['small', 'large']:
        image, wrong_image = extract_images(sample=sample, index=index, size=size)
        extracted_fields.append(image)
        extracted_fields.append(wrong_image)

    txt = np.frombuffer(
        sample['text'].numpy()[index], dtype=np.float32
    ).reshape(-1, embedding_size)
    emb_idxs = np.random.choice(txt.shape[0], size=num_embeddings_to_sample, replace=False)
    extracted_fields.append(np.mean(txt[emb_idxs, :], axis=0))

    return extracted_fields

def extract_images(sample: dict, index: int, size: str):
    """ NOTE: If include_wrong == False, then we return `None` in place """
    if size in ['small', 'large']:
        image = Image.open(io.BytesIO(sample[f'image_{size}'].numpy()[index]))
        wrong_image = Image.open(io.BytesIO(sample[f'wrong_image_{size}'].numpy()[index]))
        return image, wrong_image
    else:
        raise Exception(f'There are only two sizes: small and large. Received: {size}')

def tensors_from_sample(sample: dict, batch_size: int, text_embedding_size: int, num_samples: int, augment: bool):
    """ Extract and format the input samples such that they are ready to be fed into the model """
    image_small_tensor = []
    wrong_image_small_tensor = []
    image_large_tensor = []
    wrong_image_large_tensor = []
    text_tensor = []
    for i in range(batch_size):
        image_small, wrong_image_small, image_large, wrong_image_large, text = extract_image_with_text(
            sample=sample,
            index=i,
            embedding_size=text_embedding_size,
            num_embeddings_to_sample=num_samples
        )
        image_small = np.asarray(image_small, dtype='float32')
        wrong_image_small = np.asarray(wrong_image_small, dtype='float32')
        image_large = np.asarray(image_large, dtype='float32')
        wrong_image_large = np.asarray(wrong_image_large, dtype='float32')
        text = np.asarray(text, dtype='float32')

        if augment:
            image_small = transform_image(image_small)
            wrong_image_small = transform_image(wrong_image_small)
            image_large = transform_image(image_large)
            wrong_image_large = transform_image(wrong_image_large)

        image_small_tensor.append(image_small)
        wrong_image_small_tensor.append(wrong_image_small)
        image_large_tensor.append(image_large)
        wrong_image_large_tensor.append(wrong_image_large)
        text_tensor.append(text)

    image_small_tensor = np.asarray(image_small_tensor, dtype='float32')
    wrong_image_small_tensor = np.asarray(wrong_image_small_tensor, dtype='float32')
    image_large_tensor = np.asarray(image_large_tensor, dtype='float32')
    wrong_image_large_tensor = np.asarray(wrong_image_large_tensor, dtype='float32')
    text_tensor = np.asarray(text_tensor, dtype='float32')

    assert image_small_tensor.shape == wrong_image_small_tensor.shape, \
        'Small real ({}) and wrong ({}) images must have the same dimensions'.format(
            image_small_tensor.shape, image_small_tensor.shape
        )
    assert image_large_tensor.shape == wrong_image_large_tensor.shape, \
        'Large real ({}) and wrong ({}) images must have the same dimensions'.format(
            image_large_tensor.shape, wrong_image_large_tensor.shape
        )
    return image_small_tensor, wrong_image_small_tensor, image_large_tensor, wrong_image_large_tensor, text_tensor
