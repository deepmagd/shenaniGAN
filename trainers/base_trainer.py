import os
import tensorflow as tf
from utils.logger import MetricsLogger


class Trainer(object):
    def __init__(self, model, batch_size, save_location,
                 show_progress_bar=True):
        """ Initialise the model trainer
            Arguments:
            model: models.ConditionalGAN
                The model to train
            batch_size: int
                The number of samples per mini-batch
            save_location: str
                The directory in which to save all
                results from training the model.
        """
        self.show_progress_bar = show_progress_bar
        self.model = model
        self.batch_size = batch_size
        self.save_dir = save_location
        self.train_logger = MetricsLogger(os.path.join(self.save_dir, 'train.csv'))

    def __call__(self, data_loader, num_epochs):
        """ Trains the model.
            Arguments:
            data_loader: DirectoryIterator
                Yields tuples (x, y)
            num_epochs: int
                Number of epochs to train the model for.
        """
        for epoch_num in range(num_epochs):
            metrics = self.train_epoch(data_loader, epoch_num)
            metrics['epoch'] = epoch_num
            print(f'Metrics: {metrics}')
            self.train_logger(metrics)
            self.save_model(epoch_num)
        self.train_logger.close()

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def save_model(self, epoch_num):
        # NOTE: Hard coded for now
        dummy_noise = tf.constant(0, shape=(8, 100), dtype=tf.float32)
        dummy_image = tf.constant(0, shape=(8, 64, 64, 3), dtype=tf.float32)
        dummy_text = tf.constant(0, shape=(8, 1024), dtype=tf.float32)

        # self.model.generator(dummy_text, dummy_noise)
        tf.saved_model.save(self.model.generator, os.path.join(self.save_dir, f'model_{epoch_num}', 'generator', 'generator'))
        # self.model.generator.save(
        #     os.path.join(self.save_dir, f'model_{epoch_num}', 'generator', 'generator'),
        #     save_format='tf'
        # )
        # self.model.discriminator(dummy_image, dummy_text)
        tf.saved_model.save(self.model.discriminator, os.path.join(self.save_dir, f'model_{epoch_num}', 'discriminator', 'discriminator'))
        # self.model.discriminator.save(
        #     os.path.join(self.save_dir, f'model_{epoch_num}', 'discriminator', 'discriminator'),
        #     save_format='tf'
        # )
        # self.checkpoint.save(os.path.join(self.save_dir, f'model_{epoch_num}', "ckpt"))
        # self.model.save(
        #     os.path.join(self.save_dir, f'model_{epoch_num}'),
        #     save_format='tf'
        # )
