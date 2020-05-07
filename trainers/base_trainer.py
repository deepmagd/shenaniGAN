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
        self.val_logger = MetricsLogger(os.path.join(self.save_dir, 'val.csv'))

    def __call__(self, train_loader, val_loader, num_epochs):
        """ Trains the model.
            Arguments:
            train_loader: DirectoryIterator
                Yields tuples (x, y)
            num_epochs: int
                Number of epochs to train the model for.
        """
        for epoch_num in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch_num)
            train_metrics['epoch'] = epoch_num
            print(f'Metrics: {train_metrics}')
            self.train_logger(train_metrics)
            # Validation
            val_metrics = self.val_epoch(val_loader, epoch_num)
            val_metrics['epoch'] = epoch_num
            print(f'Metrics: {val_metrics}')
            self.val_logger(val_metrics)
            # Save
            self.save_model(epoch_num)

        self.train_logger.close()
        self.val_logger.close()

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def val_epoch(self, val_loader):
        pass

    def save_model(self, epoch_num):
        tf.saved_model.save(self.model.generator, os.path.join(self.save_dir, f'model_{epoch_num}', 'generator', 'generator'))
        tf.saved_model.save(self.model.discriminator, os.path.join(self.save_dir, f'model_{epoch_num}', 'discriminator', 'discriminator'))
