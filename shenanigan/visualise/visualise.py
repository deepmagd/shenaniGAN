import os

import numpy as np
import tensorflow as tf
from typing import Optional
from PIL import Image

from shenanigan.utils.utils import mkdir, rmdir
from shenanigan.visualise.sampler import sample_data
from shenanigan.visualise.utils import concate_horizontallly
from shenanigan.utils.data_helpers import transform_image


def compare_generated_to_real(
    dataloader,
    num_images: int,
    noise_size: int,
    model: tf.keras.Model,
    save_location: str,
    img_size: int,
    subsequent_model: Optional[tf.keras.Model] = None,
):
    """ For a given number of images, generate the stackGAN stage 1 output by randomly sampling a dataloader.
        The generated images and the real original are saved side-by-side in the save_location.
    """
    rmdir(save_location)
    mkdir(save_location)

    noise_list = [np.random.normal(0, 1, (1, noise_size)).astype('float32') for idx in range(num_images)]
    samples = sample_data(dataloader, num_samples=num_images, img_size=img_size)
    real_tensors, real_embeddings = zip(*samples)
    stage1_tensors = [
        model.generator([embedding, noise], training=False)[0] for embedding, noise in zip(real_embeddings, noise_list)
    ]

    real_images = format_as_images(real_tensors, is_real=True)
    stage1_images = format_as_images(stage1_tensors, is_real=False)

    if subsequent_model is not None:
        stage2_tensors = [
            subsequent_model.generator([generated_image, embedding], training=False)[0]
            for generated_image, embedding in zip(stage1_tensors, real_embeddings)
        ]
        stage2_images = format_as_images(stage2_tensors, is_real=False)
        for i, (real_image, stage1_image, stage2_image) in enumerate(zip(real_images, stage1_images, stage2_images)):
            image = concate_horizontallly(real_image, stage1_img=stage1_image, stage2_img=stage2_image)
            image.save(os.path.join(save_location, f'fake-vs-real-{i}.png'))
    else:
        for i, (real_image, stage1_image) in enumerate(zip(real_images, stage1_images)):
            image = concate_horizontallly(real_image, stage1_img=stage1_image)
            image.save(os.path.join(save_location, f'fake-vs-real-{i}.png'))


def format_as_images(tensors: list, is_real: bool = False):
    image_list = []
    for tensor in tensors:
        if is_real:
            tensor = transform_image(tensor)
        if len(tensor.shape) == 4:
            tensor = tf.squeeze(tensor, axis=0)
        image = (tensor.numpy() + 1) * 255.0 / 2
        image_list.append(Image.fromarray(image.astype(np.uint8)))
    return image_list
