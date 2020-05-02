import numpy as np
import os
from PIL import Image
import tensorflow as tf
from utils.utils import mkdir, rmdir
from visualise.utils import concate_horizontallly
from visualise.sampler import sample_data


def compare_generated_to_real(dataloader, num_images, conditional_emb_size, model, save_location):
    """ For a given number of images, generate the stackGAN stage 1 output by randomly sampling a dataloader.
        The generated images and the real original are saved side-by-side in the save_location.
    """
    rmdir(save_location)
    mkdir(save_location)
    noise_list = [tf.random.normal([1, conditional_emb_size]) for idx in range(num_images)]

    samples = sample_data(dataloader, num_samples=num_images)
    real_images, real_embeddings = zip(*samples)

    fake_tensors = [
        model.generator(embedding, noise, training=False)[0] for embedding, noise in zip(real_embeddings, noise_list)
    ]

    for i, (real_image, fake_tensor) in enumerate(zip(real_images, fake_tensors)):
        fake_image = tf.squeeze(fake_tensor, axis=0).numpy() * 255
        image = concate_horizontallly(real_image, Image.fromarray(fake_image.astype(np.uint8)))
        image.save(os.path.join(save_location, f'fake-vs-real-{i}.png'))
