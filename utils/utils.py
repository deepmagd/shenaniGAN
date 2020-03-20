from itertools import repeat
import json
import os
import pandas as pd
import pickle
import tensorflow as tf
import yaml


def format_file_name(image_source_dir, file_name):
    """ Format the file name (to make it compatible with windows) and uses
        utf-8 encoding.
    """
    if os.name == 'nt':
        # Check to see if running in Windows
        file_name = format_for_windows(file_name)
    return os.path.join(image_source_dir, '{}.jpg'.format(file_name)).encode('utf-8')

def read_pickle(path_to_pickle):
    """ Read a pickle file in latin encoding and return the contents """
    with open(path_to_pickle, 'rb') as pickle_file:
        content = pickle.load(pickle_file, encoding='latin1')
    return content

def chunk_list(unchuncked_list, samples_per_shard, end_point):
    """ Split a list up into evenly sized chunks / shards.
        Arguments:
            unchuncked_list: List
                A one-dimensional list
            samples_per_shard: int
                The number of samples to save in a shard
            end_point: int
                The last index that can be equally chunked
    """
    chunked_list = list(chunks(unchuncked_list[:end_point], samples_per_shard))
    chunked_list[-1].extend(unchuncked_list[end_point:])
    return chunked_list

def chunks(unchuncked_list, n):
    """ Yield successive n-sized chunks from a list. """
    for i in range(0, len(unchuncked_list), n):
        yield unchuncked_list[i:i + n]

def get_default_settings(settings_file='settings.yml'):
    with open(settings_file) as f:
        return yaml.safe_load(f)

def sample_normal(mean, log_var):
    """ Use the reparameterization trick to sample a normal distribution.
        Arguments
        mean : Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        log_var : Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
    """
    std = tf.math.exp(0.5 * log_var)
    epsilon = tf.random.normal(std)
    return mean + std * epsilon

def product_list(num_list):
    """ A helper function to simply find the
        product of all elements in the list.
    """
    product = 1
    for dim in num_list:
        product *= dim
    return product

def mkdir(directory):
    """ Create directory if it does not exist. """
    try:
        os.makedirs(directory)
    except OSError:
        pass

def save_options(options, save_dir):
    """ Save all options to JSON file.
        Arguments:
            options: An object from argparse
            save_dir: String location to save the options
    """
    opt_dict = {}
    for option in vars(options):
        opt_dict[option] = getattr(options, option)

    mkdir(save_dir)
    opts_file_path = os.path.join(save_dir, 'opts.json')
    with open(opts_file_path, 'w') as opt_file:
        json.dump(opt_dict, opt_file)

def format_for_windows(path_string):
    """ Convert to windows path by replacing `/` with `\` """
    return str(str(path_string).replace('/', '\\'))

def num_tfrecords_in_dir(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(name) and name.endswith('.tfrecord')])

def load_tabular_data(tabular_xray_path='data/CheXpert-v1.0-small/train.csv'):
    tab_xray_df = pd.read_csv(tabular_xray_path).fillna('nan')
    return tab_xray_df

def build_encoding_map(column):
    encoding_map = {}
    unique_value_list = column.unique().tolist()
    for idx, unique_value in enumerate(unique_value_list):
        encoding_map[unique_value] = idx
    return encoding_map

def encode_tabular_data(tab_xray_df):
    encoded_df = pd.DataFrame()
    for column in tab_xray_df:
        if column != 'Path':
            encoding_map = build_encoding_map(tab_xray_df[column])
            if not column in tab_xray_df:
                print(f'{elem} is not in {encoding_map}')
            encoded_column =  list(map(
                lambda elem, encoding_map: encoding_map[elem],
                tab_xray_df[column],
                repeat(encoding_map)
            ))
            encoded_df[column] = encoded_column
        else:
            encoded_df[column] = tab_xray_df[column]
    return encoded_df
