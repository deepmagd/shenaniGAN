import numpy as np
from random import randint
from utils.data_helpers import extract_image_with_text


NUM_EMBEDDINGS_TO_SAMPLE = 4


def sample_data(data_loader, num_samples):
    sample_list = []
    sample_fn = select_sample_fn(data_loader)
    for sample_idx in range(num_samples):
        sample_list.append(sample_fn(data_loader))
    return sample_list

def select_sample_fn(data_loader):
    """ A function to determine which sampling function to select """
    if data_loader.dataset_object.type == 'images-with-captions':
        return sample_img_with_captions
    else:
        raise NotImplementedError(f'Sampling dataset stype: {data_loader.dataset_object.type} is not ready yet')

def sample_img_with_captions(data_loader):
    """ A function which samples from the images-with-captions dataset.
        We return the image and caption (embedding) as a tuple
    """
    sample = next(iter(data_loader.parsed_subset))
    sample_batch_size = len(sample['text'].numpy())
    random_idx = randint(0, sample_batch_size - 1)
    img, txt = extract_image_with_text(
        sample=sample,
        index=random_idx,
        embedding_size=1024,
        num_embeddings_to_sample=NUM_EMBEDDINGS_TO_SAMPLE
    )
    if len(txt.shape) == 1:
        txt = txt[np.newaxis, :]
    return (img, txt)
