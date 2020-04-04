import io

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import trange

from trainers.base_trainer import Trainer


class TextToImageTrainer(Trainer):
    """ Trainer which feeds in text as input to the GAN to generate images """
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
        # epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch_num + 1),
                      leave=False,
                      disable=not self.show_progress_bar
        )

        with trange(len(train_loader), **kwargs) as t:
            for counter, sample in enumerate(train_loader.parsed_subset):
                # For loop simply to demonstrate the number of images in the batch
                # for i in range(self.batch_size):
                #     image_tensor = np.asarray(Image.open(io.BytesIO(sample['image_raw'].numpy()[i])))
                #     print(counter, image_tensor.shape)

                # image_tensor = np.asarray(Image.open(io.BytesIO(sample['image_raw'].numpy())))
                # print(counter, image_tensor.shape)
                # For tabular: text_tensor = np.frombuffer(sample['text'].numpy())
                # For Caption: text_tensor = np.frombuffer(sample['text'].numpy(), dtype=np.float32).reshape(10, 1024)
                # print(text_tensor)
                # name = sample['name'].numpy().decode("utf-8")
                # label = sample['label'].numpy()
                batch_size = 1 # TODO account for batch size in TFRecords
                image_tensor = np.expand_dims(np.asarray(Image.open(io.BytesIO(sample['image_raw'].numpy())), dtype=np.float32), axis=0)
                text_tensor = np.frombuffer(sample['text'].numpy(), dtype=np.float32).reshape(batch_size, 10, 1024) # TODO make dynamic
                label = sample['label'].numpy()
                print(counter, image_tensor.shape)
                # name = sample['name'].numpy().decode("utf-8")
                # For tabular: text_tensor = np.frombuffer(sample['text'].numpy())
                # For Caption: text_tensor = np.frombuffer(sample['text'].numpy(), dtype=np.float32).reshape(10, 1024)

                noise_z = tf.random.normal([batch_size, 100]) # TODO make 100 dynamic

                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                    smoothed_embedding, mean, log_sigma = self.model.conditional_augmentation(text_tensor)
                    embedding_z = tf.concat([smoothed_embedding, noise_z], 1)
                    fake_images = self.model.generator(embedding_z)

                    assert fake_images.shape == image_tensor.shape, \
                        'Real ({}) and fakes ({}) images must have the same dimensions'.format(
                            image_tensor.shape, fake_images.shape
                        )

                    real_predictions = self.model.discriminator(image_tensor, text_tensor)
                    fake_predictions = self.model.discriminator(fake_images, text_tensor)

                    assert real_predictions.shape == fake_predictions.shape, \
                        'Predictions for real ({}) and fakes ({}) images must have the same dimensions'.format(
                            real_predictions.shape, fake_predictions.shape
                        )

                    generator_loss = self.model.generator.loss(fake_predictions, mean, log_sigma)
                    discriminator_loss = self.model.discriminator.loss(real_predictions, fake_predictions)

                # Update gradients
                generator_gradients = generator_tape.gradient(generator_loss, self.model.generator.trainable_variables)
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.model.discriminator.trainable_variables)

                self.model.generator.optimiser.apply_gradients(
                    zip(generator_gradients, self.model.generator.trainable_variables)
                )
                self.model.discriminator.optimiser.apply_gradients(
                    zip(discriminator_gradients, self.model.discriminator.trainable_variables)
                )
                # Update tqdm
                t.set_postfix(generator_loss=generator_loss, discriminator_loss=discriminator_loss)
                t.update()
