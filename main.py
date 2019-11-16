import argparse
from dataloaders.dataloaders import DATASETS, create_dataloaders
import sys
import tensorflow as tf
from utils.utils import get_default_settings


SETTINGS_FILE = 'settings.yaml'


def parse_arguments(args_to_parse):

    default_settings = get_default_settings(SETTINGS_FILE)

    descr = 'Tensorflow 2 implementation of StackGAN for generating x-ray images'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument(
        'name', type=str, help="The name of the model - used for saving and loading."
    )
    general.add_argument(
        '-d', '--dataset', help="Path to training data.", default=default_settings['dataset'], choices=DATASETS
    )

    return parser.parse_args(args_to_parse)

def main(args):
    train_loader, cv_loader, test_loader = create_dataloaders(args.dataset)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
