import argparse
from dataloaders.dataloaders import create_dataloaders
from math import floor
from models.conditional_gans import StackGAN1
from trainers.trainers import get_trainer
import os
import sys
import tensorflow as tf
from utils.data_helpers import sample_real_images, show_image_list
from utils.datasets import DATASETS, get_dataset
from utils.utils import get_default_settings, save_options


SETTINGS_FILE = 'settings.yaml'
RESULTS_ROOT = 'results'


def parse_arguments(args_to_parse):
    """ Parse CLI arguments """
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

    data = parser.add_argument_group('Data settings')
    data.add_argument(
        '--samples-per-shard', type=int, default=default_settings['samples_per_shard'],
        help="The number of samples to save in each TFRecord shard."
    )

    training = parser.add_argument_group('Training settings')
    training.add_argument(
        '--use-pretrained', action='store_true', default=False,
        help='Load a pretrained model for inference'
    )
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

    visualisation = parser.add_argument_group('Visualisation settings')
    visualisation.add_argument(
        '--visualise', action='store_true', default=False,
        help='Run visualisations after loading / training the model'
    )
    visualisation.add_argument(
        '-i', '--images-to-generate', type=int, default=default_settings['num_images_to_generate'],
        help='The number of images to generate and visualise.'
    )
    visualisation.add_argument(
        '-c', '--images-to-classify', type=int, default=default_settings['num_images_to_classify'],
        help='The number of images to classify and visualise.'
    )
    training.add_argument(
        '-s', '--target-size', type=int, default=default_settings['target_size'],
        help='The target dimension for the training data. \
              E.g. --target-size = 256 results in a (256, 256) image'
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
    # TODO: Check and see how many latent dims should be used and insert CLI argument
    if args.use_pretrained:
        model = tf.saved_model.load(results_dir)
    else:
        model = StackGAN1(
            img_size=dataset_dims,
            num_latent_dims=100,
            kernel_size=args.kernel_size,
            num_filters=args.num_filters,
            reshape_dims=[args.target_size//4, args.target_size//4, args.num_filters] # NOTE divide by 4 based on 2 upsample layers in generator, will need to adjust this number
                                                                                      # based on the number of layers
        )

        trainer_class = get_trainer(args.dataset_name)
        trainer = trainer_class(
            model=model,
            batch_size=args.batch_size,
            save_location=results_dir
        )
        trainer(train_loader, num_epochs=args.num_epochs)

    if args.visualise:
        # TODO: Check if the model is in eval mode
        # Visualise fake images
        fake_images = model.generate_images(num_images=args.images_to_generate)
        show_image_list(fake_images, save_dir=results_dir)

        # Classify images
        num_images_to_classify = args.images_to_classify
        num_fakes_to_generate = floor(num_images_to_classify / 2)
        num_real_images = num_images_to_classify - num_fakes_to_generate

        dataset_object = get_dataset(args.dataset_name)
        real_images = sample_real_images(num_images=num_real_images, dataset_object=dataset_object)
        fake_images = model.generate_images(num_images=num_fakes_to_generate)

        images = real_images + fake_images
        predictions = model.classify_images(images)
        print('Predictions: {}'.format([prediction.numpy() for prediction in predictions]))
        print('The first {} are real, while the lst {} are fake'.format(len(real_images), len(fake_images)))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
