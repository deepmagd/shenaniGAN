import io
import numpy as np
from PIL import Image
from random import randint


NUM_EMBEDDINGS_TO_SAMPLE = 4


def sample_data(data_loader, num_samples):
    sample_list = []
    sample_fn = select_sample_fn(data_loader)
    for sample_idx in range(num_samples):
        sample_list.append(sample_fn(data_loader))
    return sample_list

def select_sample_fn(data_loader):
    """ """
    print(f'data_loader.dataset_object.type: {data_loader.dataset_object.type}')
    if data_loader.dataset_object.type == 'images-with-captions':
        return sample_img_with_captions
    else:
        raise NotImplementedError(f'Sampling dataset stype: {data_loader.dataset_object.type} is not ready yet')

def sample_img_with_captions(data_loader):
    """ """
    sample = next(iter(data_loader.parsed_subset))
    sample_batch_size = len(sample['text'].numpy())
    random_idx = randint(0, sample_batch_size - 1)
    img = Image.open(io.BytesIO(sample['image_raw'].numpy()[random_idx]))
    txt = np.frombuffer(
        sample['text'].numpy()[random_idx], dtype=np.float32
    ).reshape(-1, 1024)[:NUM_EMBEDDINGS_TO_SAMPLE - 1, :]
    txt = np.mean(txt, axis=0, keepdims=True)
    return (img, txt)
