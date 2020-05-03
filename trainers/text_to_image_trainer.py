import io
from random import randint
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import tensorflow as tf
from tqdm import trange
from trainers.base_trainer import Trainer
from utils.data_helpers import transform_image
from utils.utils import extract_image_with_text


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
        self.conditional_emb_size = kwargs.get('conditional_emb_size')
        self.augment = kwargs.get('augment')

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        acc_generator_loss = 0
        acc_discriminator_loss = 0
        kwargs = dict(
            desc="Epoch {}".format(epoch_num + 1),
            leave=False,
            disable=not self.show_progress_bar
        )
        with trange(len(train_loader), **kwargs) as t:
            for batch_idx, sample in enumerate(train_loader.parsed_subset):
                image_tensor = []
                wrong_image_tensor = []
                text_tensor = []
                batch_size = len(sample['text'].numpy())
                for i in range(batch_size):
                    img, wrong_img, txt = extract_image_with_text(
                        sample=sample,
                        index=i,
                        embedding_size=1024,
                        num_embeddings_to_sample=self.num_samples,
                        include_wrong=True
                    )
                    img = np.asarray(img, dtype='float32')
                    wrong_img = np.asarray(wrong_img, dtype='float32')
                    if self.augment:
                        img = transform_image(img)
                        wrong_img = transform_image(wrong_img)
                    image_tensor.append(img)
                    wrong_image_tensor.append(wrong_img)
                    text_tensor.append(txt)
                image_tensor = np.asarray(image_tensor, dtype='float32')
                wrong_image_tensor = np.asarray(wrong_image_tensor, dtype='float32')
                text_tensor = np.asarray(text_tensor, dtype='float32')

                assert image_tensor.shape == wrong_image_tensor.shape, \
                    'Real ({}) and wrong ({}) images must have the same dimensions'.format(
                        image_tensor.shape, wrong_image_tensor.shape
                    )

                noise_z = tf.random.normal([batch_size, self.conditional_emb_size])

                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    fake_images, mean, log_sigma = self.model.generator(text_tensor, noise_z, training=True)

                    assert fake_images.shape == image_tensor.shape, \
                        'Real ({}) and fakes ({}) images must have the same dimensions'.format(
                            image_tensor.shape, fake_images.shape
                        )

                    real_predictions = self.model.discriminator(image_tensor, text_tensor, training=True)
                    wrong_predictions = self.model.discriminator(wrong_image_tensor, text_tensor, training=True)
                    fake_predictions = self.model.discriminator(fake_images.numpy(), text_tensor, training=True)

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

                # Accumulate losses
                acc_generator_loss += generator_loss
                acc_discriminator_loss += discriminator_loss

                # if batch_idx == 20:
                #     break

        return {
            'generator_loss': np.asscalar(acc_generator_loss.numpy()) / (batch_idx + 1),
            'discriminator_loss': np.asscalar(acc_discriminator_loss.numpy()) / (batch_idx + 1)
        }
