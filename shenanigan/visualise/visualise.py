import os

import numpy as np
import tensorflow as tf
from PIL import Image

from shenanigan.utils.utils import mkdir, rmdir
from shenanigan.visualise.sampler import sample_data
from shenanigan.visualise.utils import concate_horizontallly


def compare_generated_to_real(dataloader, num_images, noise_size, model, save_location):
    """ For a given number of images, generate the stackGAN stage 1 output by randomly sampling a dataloader.
        The generated images and the real original are saved side-by-side in the save_location.
    """
    rmdir(save_location)
    mkdir(save_location)

    noise_list = [np.random.normal(0, 1, (1, noise_size)).astype('float32') for idx in range(num_images)]
    samples = sample_data(dataloader, num_samples=num_images)
    real_images, real_embeddings = zip(*samples)

    fake_tensors = [
        model.generator([embedding, noise], training=False)[0] for embedding, noise in zip(real_embeddings, noise_list)
    ]

    for i, (real_image, fake_tensor) in enumerate(zip(real_images, fake_tensors)):
        fake_image = (tf.squeeze(fake_tensor, axis=0).numpy() + 1) * 255. / 2
        image = concate_horizontallly(real_image, Image.fromarray(fake_image.astype(np.uint8)))
        image.save(os.path.join(save_location, f'fake-vs-real-{i}.png'))
