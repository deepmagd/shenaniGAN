import os

import numpy as np
import tensorflow as tf
from PIL import Image

from shenanigan.utils.utils import mkdir, rmdir
from shenanigan.visualise.sampler import sample_data
from shenanigan.visualise.utils import concate_horizontallly
from shenanigan.utils.data_helpers import transform_image


def compare_generated_to_real(dataloader, num_images, noise_size, model, save_location, img_size, subsequent_model=None):
    """ For a given number of images, generate the stackGAN stage 1 output by randomly sampling a dataloader.
        The generated images and the real original are saved side-by-side in the save_location.
    """
    rmdir(save_location)
    mkdir(save_location)

    noise_list = [np.random.normal(0, 1, (1, noise_size)).astype('float32') for idx in range(num_images)]
    samples = sample_data(dataloader, num_samples=num_images, img_size=img_size)
    real_tensors, real_embeddings = zip(*samples)
    fake_tensors = [
        model.generator([embedding, noise], training=False)[0] for embedding, noise in zip(real_embeddings, noise_list)
    ]

    if subsequent_model is not None:
        fake_tensors = [
            subsequent_model.generator([generated_image, embedding], training=False)[0] for generated_image, embedding in zip(fake_tensors, real_embeddings)
        ]

    for i, (real_tensor, fake_tensor) in enumerate(zip(real_tensors, fake_tensors)):
        real_tensor = transform_image(real_tensor)
        if len(fake_tensor.shape) == 4:
            fake_tensor = tf.squeeze(fake_tensor, axis=0)
        fake_image = (fake_tensor.numpy() + 1) * 255. / 2
        real_image = (real_tensor.numpy() + 1) * 255. / 2
        image = concate_horizontallly(Image.fromarray(real_image.astype(np.uint8)), Image.fromarray(fake_image.astype(np.uint8)))
        image.save(os.path.join(save_location, f'fake-vs-real-{i}.png'))
