import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from shenanigan.dataloaders import create_dataloaders
from shenanigan.models.stackgan import run_stackgan
from shenanigan.utils import get_default_settings, save_options
from shenanigan.utils.datasets import DATASETS
from shenanigan.models.inception import run_inception

RESULTS_ROOT = 'results'
SEED = 1234

tf.random.set_seed(SEED)
np.random.seed(SEED)

MODELS = ['stackgan', 'inception']

def parse_arguments(args_to_parse):
    """ Parse CLI arguments """

    descr = 'shenaniGAN: An implementation of different multi-modal and conditional GANs'
    parser = argparse.ArgumentParser(description=descr)

    general = parser.add_argument_group('General settings')
    general.add_argument(
        'name', type=str, help="The name of the model - used for saving and loading."
    )
    general.add_argument(
        '-m', '--model', type=str, help="Which model architecture to use.",
        choices=MODELS
    )
    general.add_argument(
        '-d', '--dataset-name', type=str, help="Name of the dataset to use during training.",
        choices=DATASETS
    )
    general.add_argument(
        '--use-pretrained', action='store_true', default=False,
        help='Load a pretrained model for inference'
    )
    general.add_argument(
        '--visualise', action='store_true', default=False,
        help='Run visualisations after loading / training the model'
    )
    general.add_argument(
        '--evaluate', action='store_true', default=False,
        help='Run evaluation metrics'
    )

    stackgan = parser.add_argument_group('StackGAN settings')
    stackgan.add_argument(
        '-s', '--stage', type=int, choices=[1, 2], required=False,
        help='Whether to train stage 1 or 2.'
    )

    parsed_args = parser.parse_args(args_to_parse)
    return parsed_args

def main(args):
    default_settings = get_default_settings(f'shenanigan/models/{args.model}/settings.yaml')

    if args.model == 'stackgan':
        train_loader, val_loader, small_image_dims, _ = create_dataloaders(args.dataset_name, default_settings['common']['batch_size'])
        results_dir = os.path.join(RESULTS_ROOT, args.name, f'stage-{args.stage}')
        save_options(options=args, save_dir=results_dir)
        run_stackgan(train_loader, val_loader, small_image_dims, results_dir, default_settings, args.name, args.stage, args.use_pretrained, args.visualise, args.evaluate)
    elif args.model == 'inception':
        run_inception(args.name, args.dataset_name, default_settings)
    else:
        raise NotImplementedError(f"No implementation for model '{args.model}'")


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
