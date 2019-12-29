import tensorflow as tf
from tqdm import trange


class Trainer():
    def __init__(self, model, show_progress_bar=True):
        """ Initialise the model trainer
            Arguments:
            model: models.ConditionalGAN
                The model to train
        """
        self.show_progress_bar = show_progress_bar
        self.model = model

    def __call__(self, data_loader, num_epochs):
        """ Trains the model.
            Arguments:
            data_loader: DirectoryIterator
                Yields tuples (x, y)
            num_epochs: int
                Number of epochs to train the model for.
        """
        for epoch_num in range(num_epochs):
            self.train_epoch(data_loader, epoch_num)

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        # epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch_num + 1),
                      leave=False,
                      disable=not self.show_progress_bar
        )
        with trange(len(train_loader), **kwargs) as t:
            for real_images, one_hot_labels in train_loader:
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
