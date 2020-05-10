import os

import tensorflow as tf

from shenanigan.callbacks import LearningRateDecay
from shenanigan.utils import extract_epoch_num
from shenanigan.utils.logger import LogPlotter
from shenanigan.visualise import compare_generated_to_real

from . import StackGAN1, StackGAN2
from .utils import get_trainer


def load_model(settings, image_dims, results_dir, stage, epoch_num=-1):
    model = StackGAN1(
        img_size=image_dims,
        lr_g=settings['stage1']['generator']['learning_rate'],
        lr_d=settings['stage1']['discriminator']['learning_rate'],
        conditional_emb_size=settings['stage1']['conditional_emb_size'],
        w_init=tf.random_normal_initializer(stddev=0.02),
        bn_init=tf.random_normal_initializer(1., 0.02)
    )
    if stage == 2:
        results_dir = results_dir.replace('stage-2', 'stage-1')

    if epoch_num == -1:
        # Find last checkpoint
        epoch_num = extract_epoch_num(results_dir)

    pretrained_dir = os.path.join(results_dir, f'model_{epoch_num}')
    model.generator = tf.saved_model.load(os.path.join(pretrained_dir, 'generator', 'generator'))
    model.discriminator = tf.saved_model.load(os.path.join(pretrained_dir, 'discriminator', 'discriminator'))
    return model

def run(train_loader, val_loader, small_image_dims, results_dir, settings, stage, use_pretrained=False, visualise=False):
    lr_decay = LearningRateDecay(
        decay_factor=settings['callbacks']['learning_rate_decay']['decay_factor'],
        every_n=settings['callbacks']['learning_rate_decay']['every_n']
    )

    # Create the model
    if stage == 1 and use_pretrained:
        model = load_model(settings, small_image_dims, results_dir, stage, settings['stage1']['epoch_num'])

    elif stage == 1:
        model = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings['stage1']['generator']['learning_rate'],
            lr_d=settings['stage1']['discriminator']['learning_rate'],
            conditional_emb_size=settings['stage1']['conditional_emb_size'],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1., 0.02)
        )

        trainer_class = get_trainer(stage)
        trainer = trainer_class(
            model=model,
            batch_size=settings['common']['batch_size'],
            save_location=results_dir,
            save_every=settings['stage1']['save_every_n_epochs'],
            save_best_after=settings['stage1']['save_best_after_n_epochs'],
            callbacks=[lr_decay],
            num_samples=settings['stage1']['num_samples'],
            noise_size=settings['stage1']['noise_size'],
            augment=settings['stage1']['augment']
        )
        trainer(train_loader, val_loader, num_epochs=settings['stage1']['num_epochs'])

        # Plot metrics
        plotter = LogPlotter(results_dir)
        plotter.learning_curve()

    elif stage == 2 and use_pretrained:
        raise NotImplementedError('We have not yet provided the ability to load pretrained stage 2 models')

    elif stage == 2:
        model_stage2 = StackGAN2(
            img_size=small_image_dims,
            lr_g=settings['stage2']['generator']['learning_rate'],
            lr_d=settings['stage2']['discriminator']['learning_rate'],
            conditional_emb_size=settings['stage2']['conditional_emb_size'],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1., 0.02)
        )
        model_stage1 = load_model(settings, small_image_dims, results_dir, stage, settings['stage1']['epoch_num'])

        trainer_class = get_trainer(stage)
        trainer = trainer_class(
            model=model_stage2,
            batch_size=settings['common']['batch_size'],
            save_location=results_dir,
            save_every=settings['stage2']['save_every_n_epochs'],
            save_best_after=settings['stage2']['save_best_after_n_epochs'],
            callbacks=[lr_decay],
            num_samples=settings['stage2']['num_samples'],
            noise_size=settings['stage1']['noise_size'],
            augment=settings['stage2']['augment'],
            stage_1_generator=model_stage1.generator
        )
        trainer(train_loader, val_loader, num_epochs=settings['stage2']['num_epochs'])

        # Plot metrics
        plotter = LogPlotter(results_dir)
        plotter.learning_curve()

    if visualise:
        # TODO: Check if the model is in eval mode
        # Visualise fake images from training set
        compare_generated_to_real(
            dataloader=train_loader,
            num_images=settings['visualisation']['images_to_generate'],
            noise_size=settings['stage1']['noise_size'],
            model=model,
            save_location=os.path.join(results_dir, 'viz')
        )
