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
        self.train_logger = MetricsLogger(os.path.join(self.save_dir, 'train.log'))

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
            print(f'Metrics: {metrics}')
            self.train_logger(metrics)
            self.save_model(epoch_num)

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def save_model(self, epoch_num):
        tf.saved_model.save(self.model, os.path.join(self.save_dir, f'model_{epoch_num}'))
