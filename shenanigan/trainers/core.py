import os

import tensorflow as tf

from shenanigan.utils.logger import MetricsLogger
from shenanigan.utils.model_helpers import Checkpointer


class Trainer(object):
    def __init__(self, model, batch_size, save_location,
                 save_every, save_best_after, callbacks=None, use_pretrained=False, show_progress_bar=True):
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
        self.model = model
        self.batch_size = batch_size
        self.save_dir = save_location
        self.save_every = save_every
        self.save_best_after = save_best_after
        self.train_logger = MetricsLogger(os.path.join(self.save_dir, 'train.csv'), use_pretrained)
        self.val_logger = MetricsLogger(os.path.join(self.save_dir, 'val.csv'), use_pretrained)
        self.show_progress_bar = show_progress_bar
        self.callbacks = callbacks if callbacks is not None else []
        self.use_pretrained = use_pretrained
        self.minimum_loss = None
        # NOTE using generator loss for now, should be validation though
        self.tracking_metric = 'generator_loss'
        self.save_best_checkpointer = Checkpointer(
            model=self.model,
            save_dir=os.path.join(self.save_dir, 'ckpts_best'),
            max_keep=3
        )
        self.save_every_checkpointer = Checkpointer(
            model=self.model,
            save_dir=os.path.join(self.save_dir, 'ckpts_every'),
            max_keep=20
        )

    def __call__(self, train_loader, val_loader, num_epochs):
        """ Trains the model.
            Arguments:
            train_loader: DirectoryIterator
                Yields tuples (x, y)
            num_epochs: int
                Number of epochs to train the model for.
        """
        # NOTE the order is important, initialise model with save every as this is likely the last model saved
        self.save_best_checkpointer.restore(use_pretrained=self.use_pretrained)
        self.save_every_checkpointer.restore(use_pretrained=self.use_pretrained)

        self.minimum_loss = self.save_best_checkpointer.get_loss()

        for _ in range(num_epochs):
            # Train
            self.save_every_checkpointer.increment_epoch()
            self.save_best_checkpointer.increment_epoch()
            train_metrics = self.train_epoch(train_loader, self.save_every_checkpointer.get_epoch_num())
            train_metrics['epoch'] = int(self.save_every_checkpointer.get_epoch_num())
            print(f'Metrics: {train_metrics}')
            self.train_logger(train_metrics)
            # Validation
            val_metrics = self.val_epoch(val_loader, self.save_every_checkpointer.get_epoch_num())
            val_metrics['epoch'] = int(self.save_every_checkpointer.get_epoch_num())
            print(f'Metrics: {val_metrics}')
            self.val_logger(val_metrics)
            # update loss
            self.save_best_checkpointer.update_loss(train_metrics[self.tracking_metric])
            self.save_every_checkpointer.update_loss(train_metrics[self.tracking_metric])
            # Save
            if ((self.save_every_checkpointer.get_epoch_num() + 1) % self.save_every) == 0:
                save_path = self.save_every_checkpointer.save()
                print("Saved checkpoint for step {}: {}".format(int(self.save_every_checkpointer.get_epoch_num()), save_path))
            if self.is_best(train_metrics['generator_loss']): # NOTE this should be validation but using train for now
                save_path = self.save_best_checkpointer.save()
                print("Saved best checkpoint for step {}: {}".format(int(self.save_best_checkpointer.get_epoch_num()), save_path))
            self.run_callbacks(self.save_every_checkpointer.get_epoch_num())

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def val_epoch(self, val_loader, epoch_num):
        pass

    def is_best(self, current_loss):
        if current_loss < self.minimum_loss:
            self.minimum_loss = current_loss
            return True
        return False

    def run_callbacks(self, epoch_num: int):
        for callback in self.callbacks:
            callback(self.model.generator, epoch_num)
            callback(self.model.discriminator, epoch_num)
