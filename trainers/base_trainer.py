import tensorflow as tf


class Trainer(object):
    def __init__(self, model, save_location,
                 show_progress_bar=True):
        """ Initialise the model trainer
            Arguments:
            model: models.ConditionalGAN
                The model to train
            save_location: str
                The directory in which to save all
                results from training the model.
        """
        self.show_progress_bar = show_progress_bar
        self.model = model
        self.save_dir = save_location

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
            self.save_model()

    def train_epoch(self, train_loader, epoch_num):
        """ Training operations for a single epoch """
        pass

    def save_model(self):
        tf.saved_model.save(self.model, self.save_dir)
