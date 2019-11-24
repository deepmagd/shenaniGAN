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
            for inputs, labels in train_loader:
                # TODO: Implement this here
                print(inputs)
                # predictions = self.model(inputs, labels)
                # loss = loss_fn(predictions, labels)
                # etc

                # # Update tqdm
                # t.set_postfix(loss=loss)
                # t.update()
