import io
import numpy as np
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
                for i in range(self.batch_size):
                    image_tensor = np.asarray(Image.open(io.BytesIO(sample['image_raw'].numpy()[i])))
                    print(counter, image_tensor.shape)

                # image_tensor = np.asarray(Image.open(io.BytesIO(sample['image_raw'].numpy())))
                # print(counter, image_tensor.shape)
                # For tabular: text_tensor = np.frombuffer(sample['text'].numpy())
                # For Caption: text_tensor = np.frombuffer(sample['text'].numpy(), dtype=np.float32).reshape(10, 1024)
                # print(text_tensor)
                # name = sample['name'].numpy().decode("utf-8")
                # label = sample['label'].numpy()
