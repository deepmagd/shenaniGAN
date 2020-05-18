import os

import tensorflow as tf

from shenanigan.utils.logger import MetricsLogger


class Trainer(object):
    def __init__(self, model, batch_size, save_location,
                 save_every, save_best_after, callbacks=None, continue_training=False, show_progress_bar=True):
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
        self.train_logger = MetricsLogger(os.path.join(self.save_dir, 'train.csv'))
        self.val_logger = MetricsLogger(os.path.join(self.save_dir, 'val.csv'))
        self.minimum_val_loss = None
        self.show_progress_bar = show_progress_bar
        self.callbacks = callbacks if callbacks is not None else []
        self.continue_training = continue_training

        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0),
            discriminator=self.model.discriminator,
            generator=self.model.generator,
            g_optimizer=self.model.generator.optimizer,
            d_optimizer=self.model.discriminator.optimizer
        )
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, os.path.join(self.save_dir, 'ckpts'), max_to_keep=3)

    def __call__(self, train_loader, val_loader, num_epochs):
        """ Trains the model.
            Arguments:
            train_loader: DirectoryIterator
                Yields tuples (x, y)
            num_epochs: int
                Number of epochs to train the model for.
        """

        if self.continue_training:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            if self.ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")

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
            self.ckpt.step.assign_add(1)
            if ((epoch_num + 1) % self.save_every == 0) or \
               ((epoch_num + 1) > self.save_best_after and self.is_best(val_metrics['generator_loss'])):
                self.save_model(epoch_num)
                save_path = self.ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            self.run_callbacks(epoch_num)

        self.train_logger.close()
        self.val_logger.close()

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def val_epoch(self, val_loader, epoch_num):
        pass

    def save_model(self, epoch_num):
        tf.saved_model.save(
            self.model.generator,
            os.path.join(self.save_dir, f'model_{epoch_num}', 'generator', 'generator')
        )
        tf.saved_model.save(
            self.model.discriminator,
            os.path.join(self.save_dir, f'model_{epoch_num}', 'discriminator', 'discriminator')
        )

    def is_best(self, generator_loss):
        if self.minimum_val_loss is None or self.minimum_val_loss > generator_loss:
            self.minimum_val_loss = generator_loss
            return True
        return False

    def run_callbacks(self, epoch_num: int):
        for callback in self.callbacks:
            callback(self.model.generator, epoch_num)
            callback(self.model.discriminator, epoch_num)
