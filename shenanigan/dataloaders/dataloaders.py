import os

from shenanigan.utils.datasets import get_dataset
from shenanigan.utils.utils import num_tfrecords_in_dir


def create_dataloaders(dataset_name, batch_size):
    """ Create traing and validation set generators """
    dataset = get_dataset(dataset_name)
    if dataset.type == 'images-with-captions':
        return image_with_captions_loaders(dataset, batch_size)
    elif dataset.type == 'images-with-tabular':
        return image_with_tabular(dataset, batch_size)
    else:
        raise Exception('Unexpected dataset type: {}'.format(dataset.type))

def image_with_tabular(dataset, batch_size):
    """ Read, prepare, and present the TFRecord data for images
        alongside the corresponding tabular data
    """
    train_loader = ImageTextDataLoader(
        dataset_object=dataset,
        batch_size=batch_size,
        subset='train'
    )
    test_loader = ImageTextDataLoader(
        dataset_object=dataset,
        batch_size=batch_size,
        subset='test'
    )
    return train_loader, test_loader,  dataset.get_small_dims(), dataset.get_large_dims()

def image_with_captions_loaders(dataset, batch_size):
    """ Read, prepare, and present the TFRecord data """
    train_loader = ImageTextDataLoader(
        dataset_object=dataset,
        batch_size=batch_size,
        subset='train'
    )
    test_loader = ImageTextDataLoader(
        dataset_object=dataset,
        batch_size=batch_size,
        subset='test'
    )
    return train_loader, test_loader, dataset.get_small_dims(), dataset.get_large_dims()


class ImageTextDataLoader(object):
    """ Define a dataloader for image-text pairs """
    def __init__(self, dataset_object, batch_size, subset):
        """ Initialise a ImageTextDataLoader object for either the train
            or test subsets using a BirdsWithWordsDataset object.
        """
        # self.index = 0
        self.subset = subset
        self.batch_size = batch_size
        self.dataset_object = dataset_object
        self.parsed_subset = self.dataset_object.parse_dataset(self.subset, self.batch_size)

    def __call__(self):
        for sample in self.parsed_subset:
            yield sample

    def __len__(self):
        return num_tfrecords_in_dir(os.path.join(self.dataset_object.directory, self.subset))
