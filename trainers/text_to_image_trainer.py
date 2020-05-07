import io
from random import randint
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import tensorflow as tf
from tqdm import trange
from trainers.base_trainer import Trainer
from utils.data_helpers import tensors_from_sample


class TextToImageTrainer(Trainer):
    """ Trainer which feeds in text as input to the GAN to generate images """
    def __init__(self, model, batch_size, save_location,
                 show_progress_bar=True, **kwargs):
        """ Initialise a model trainer for iamge data.
            Arguments:
            model: models.ConditionalGAN
                The model to train
            batch_size: int
                The number of samples per mini-batch
            save_location: str
                The directory in which to save all
                results from training the model.
        """
        super().__init__(model, batch_size, save_location, show_progress_bar)
        self.num_embeddings = kwargs.get('num_embeddings')
        self.num_samples = kwargs.get('num_samples')
        self.noise_size = kwargs.get('noise_size')
        self.augment = kwargs.get('augment')

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        acc_generator_loss = 0
        acc_discriminator_loss = 0
        text_embedding_size = train_loader.dataset_object.text_embedding_dim
        kwargs = dict(
            desc="Epoch {}".format(epoch_num + 1),
            leave=False,
            disable=not self.show_progress_bar
        )
        with trange(len(train_loader), **kwargs) as t:
            for batch_idx, sample in enumerate(train_loader.parsed_subset):
                batch_size = len(sample['text'].numpy())
                image_tensor, wrong_image_tensor, text_tensor = tensors_from_sample(
                    sample,batch_size, text_embedding_size, self.num_samples, self.augment
                )

                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    noise_z = tf.random.normal((batch_size, self.noise_size))
                    fake_images, mean, log_sigma = self.model.generator([text_tensor, noise_z], training=True)

                    assert fake_images.shape == image_tensor.shape, \
                        'Real ({}) and fakes ({}) images must have the same dimensions'.format(
                            image_tensor.shape, fake_images.shape
                        )

                    real_predictions = tf.squeeze(self.model.discriminator([image_tensor, text_tensor], training=True))
                    wrong_predictions = tf.squeeze(self.model.discriminator([wrong_image_tensor, text_tensor], training=True))
                    fake_predictions = tf.squeeze(self.model.discriminator([fake_images, text_tensor], training=True))

                    assert real_predictions.shape == wrong_predictions.shape == fake_predictions.shape, \
                        'Predictions for real ({}), wrong ({}) and fakes ({}) images must have the same dimensions'.format(
                            real_predictions.shape, wrong_predictions.shape, fake_predictions.shape
                        )

                    generator_loss = self.model.generator.loss(fake_predictions, mean, log_sigma)
                    discriminator_loss = self.model.discriminator.loss(real_predictions, wrong_predictions, fake_predictions)

                # Update gradients
                generator_gradients = generator_tape.gradient(generator_loss, self.model.generator.trainable_variables)
                discriminator_gradients = discriminator_tape.gradient(
                    discriminator_loss, self.model.discriminator.trainable_variables
                )

                self.model.generator.optimiser.apply_gradients(
                    zip(generator_gradients, self.model.generator.trainable_variables)
                )
                self.model.discriminator.optimiser.apply_gradients(
                    zip(discriminator_gradients, self.model.discriminator.trainable_variables)
                )
                # Update tqdm
                t.set_postfix(generator_loss=generator_loss, discriminator_loss=discriminator_loss)
                t.update()

                # Accumulate losses over all samples
                acc_generator_loss += generator_loss
                acc_discriminator_loss += discriminator_loss

                # samples, _, _ = self.model.generator([text_tensor, noise_z], training=False)
                # temp = samples[0, :, :, :].numpy()
                # temp = ((temp + 1) / 2)#.astype(np.uint8)
                # temp[temp < 0] = 0
                # temp[temp > 1] = 1
                # matplotlib.image.imsave('gen_{}.png'.format(epoch_num), temp)

                # if batch_idx == 5:
                #     break

        return {
            'generator_loss': np.asscalar(acc_generator_loss.numpy()) / (batch_idx + 1),
            'discriminator_loss': np.asscalar(acc_discriminator_loss.numpy()) / (batch_idx + 1)
        }

    def val_epoch(self, val_loader, epoch_num):
        acc_generator_loss = 0
        acc_discriminator_loss = 0
        text_embedding_size = val_loader.dataset_object.text_embedding_dim
        kwargs = dict(
            desc="Epoch {}".format(epoch_num + 1),
            leave=False,
            disable=not self.show_progress_bar
        )
        with trange(len(val_loader), **kwargs) as t:
            for batch_idx, sample in enumerate(val_loader.parsed_subset):
                batch_size = len(sample['text'].numpy())
                image_tensor, wrong_image_tensor, text_tensor = tensors_from_sample(
                    sample, batch_size, text_embedding_size, self.num_samples, self.augment
                )
                noise_z = tf.random.normal((batch_size, self.noise_size))
                fake_images, mean, log_sigma = self.model.generator([text_tensor, noise_z], training=False)

                assert fake_images.shape == image_tensor.shape, \
                    'Real ({}) and fakes ({}) images must have the same dimensions'.format(
                        image_tensor.shape, fake_images.shape
                    )

                real_predictions = tf.squeeze(self.model.discriminator([image_tensor, text_tensor], training=False))
                wrong_predictions = tf.squeeze(self.model.discriminator([wrong_image_tensor, text_tensor], training=False))
                fake_predictions = tf.squeeze(self.model.discriminator([fake_images, text_tensor], training=False))

                assert real_predictions.shape == wrong_predictions.shape == fake_predictions.shape, \
                    'Predictions for real ({}), wrong ({}) and fakes ({}) images must have the same dimensions'.format(
                        real_predictions.shape, wrong_predictions.shape, fake_predictions.shape
                    )

                generator_loss = self.model.generator.loss(fake_predictions, mean, log_sigma)
                discriminator_loss = self.model.discriminator.loss(real_predictions, wrong_predictions, fake_predictions)

                # Update tqdm
                t.set_postfix(generator_loss=generator_loss, discriminator_loss=discriminator_loss)
                t.update()

                # Accumulate losses over all samples
                acc_generator_loss += generator_loss
                acc_discriminator_loss += discriminator_loss

                # if batch_idx == 5:
                #     break

        return {
            'generator_loss': np.asscalar(acc_generator_loss.numpy()) / (batch_idx + 1),
            'discriminator_loss': np.asscalar(acc_discriminator_loss.numpy()) / (batch_idx + 1)
        }
