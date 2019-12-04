import argparse
from dataloaders.dataloaders import create_dataloaders
from models.conditional_gans import StackGAN1
from models.trainer import Trainer
import sys
import tensorflow as tf
from utils.utils import DATASETS, get_default_settings


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
    return parser.parse_args(args_to_parse)

def main(args):
    train_loader, val_generator, dataset_dims = create_dataloaders(args)

    # Create the model
    # TODO: Check and see how many latent dims should be used and inser CLI argument
    model = StackGAN1(
        img_size=dataset_dims,
        num_latent_dims=10
    )

    # NOTE: For now, no model passed to the trainer
    trainer = Trainer(model=model)
    trainer(train_loader, num_epochs=args.num_epochs)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
