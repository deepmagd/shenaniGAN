import argparse
from dataloaders.dataloaders import create_dataloaders
from models.conditional_gans import StackGAN1
from models.trainer import Trainer
import os
import sys
import tensorflow as tf
from utils.utils import DATASETS, get_default_settings, save_options


SETTINGS_FILE = 'settings.yaml'
RESULTS_ROOT = 'results'


def parse_arguments(args_to_parse):

    default_settings = get_default_settings(SETTINGS_FILE)

    descr = 'Tensorflow 2 implementation of StackGAN for generating x-ray images'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument(
        'name', type=str, help="The name of the model - used for saving and loading."
    )
    general.add_argument(
        '-d', '--dataset-name', help="Name of the dataset to use during training.",
        default=default_settings['dataset'], choices=DATASETS
    )

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '-e', '--num-epochs', type=int, default=default_settings['num_epochs'],
        help='Maximum number of epochs to run for.'
    )
    training.add_argument(
        '-b', '--batch-size', type=int, default=default_settings['batch_size'],
        help='The number of images to use in a batch during training'
    )
    training.add_argument(
        '-k', '--kernel-size', type=int, default=default_settings['kernel_size'],
        help='Integer which indicates the size of the kernel. \
              E.g. --kernel-size = 4 results in a (4, 4) kernel'
    )
    training.add_argument(
        '-f', '--num-filters', type=int, default=default_settings['num_filters'],
        help='The number of filters to stack.'
    )
    args = parser.parse_args(args_to_parse)
    args.kernel_size = (args.kernel_size, args.kernel_size)
    return args

def main(args):
    # Save options:
    results_dir = os.path.join(RESULTS_ROOT, args.name)
    save_options(options=args, save_dir=results_dir)

    train_loader, val_generator, dataset_dims = create_dataloaders(args)

    # Create the model
    # TODO: Check and see how many latent dims should be used and inser CLI argument
    model = StackGAN1(
        img_size=dataset_dims,
        num_latent_dims=100,
        kernel_size=args.kernel_size,
        num_filters=args.num_filters,
        reshape_dims=[91, 125, args.num_filters],
        num_image_channels=dataset_dims[0]
    )

    # NOTE: For now, no model passed to the trainer
    trainer = Trainer(
        model=model,
        save_location=results_dir
    )
    trainer(train_loader, num_epochs=args.num_epochs)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
