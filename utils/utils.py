import json
import os
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

def chunk_list(unchuncked_list, num_chunks, end_point):
    """ Split a list up into evenly sized chunks / shards.
        Arguments:
            unchuncked_list: List
                A one-dimensional list
            num_chunks: int
                The number of chunks to create
            end_point: int
                The last index that can be equally chunked
    """
    chunked_list = list(chunks(unchuncked_list[:end_point], num_chunks))
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
