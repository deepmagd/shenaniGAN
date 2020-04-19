import numpy as np
import tensorflow as tf
from tqdm import trange
from trainers.base_trainer import Trainer


class ImageTrainer(Trainer):
    """ A image-to-image GAN training class """
    def __init__(self, model, batch_size, save_location,
                 show_progress_bar=True):
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
            for batch_idx, (real_images, one_hot_labels) in enumerate(train_loader):
                # Assuming that batch is the first dimension
                # and that we use a normal distribution for noise
                batch_size = real_images.shape[0]
                noise = tf.random.normal([batch_size, self.model.generator.num_latent_dims])

                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    fake_images = self.model.generator(noise)
                    assert fake_images.shape == real_images.shape, \
                        'Real ({}) and fakes ({}) images must have the same dimensions'.format(
                            real_images.shape, fake_images.shape
                        )

                    real_predictions = self.model.discriminator(real_images)
                    fake_predictions = self.model.discriminator(fake_images)

                    assert real_predictions.shape == fake_predictions.shape, \
                        'Predictions for real ({}) and fakes ({}) images must have the same dimensions'.format(
                            real_predictions.shape, fake_predictions.shape
                        )

                    generator_loss = self.model.generator.loss(fake_predictions)
                    discriminator_loss = self.model.discriminator.loss(real_predictions, fake_predictions)

                # Update gradients
                generator_gradients = generator_tape.gradient(
                    generator_loss, self.model.generator.trainable_variables
                )
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
        return {
            'generator_loss': np.asscalar(acc_generator_loss.numpy()) / (batch_idx + 1),
            'discriminator_loss': np.asscalar(acc_discriminator_loss.numpy()) / (batch_idx + 1)
        }