import numpy as np
import tensorflow as tf

from shenanigan.metrics import inception_score
from shenanigan.utils.data_helpers import tensors_from_sample


def evaluate(model, dataloader, experiment_name, num_samples, augment, noise_size):

    all_images = []
    for _, sample in enumerate(dataloader.parsed_subset):
        batch_size = len(sample['text'].numpy())
        _, _, large_images, _, _ = tensors_from_sample(
            sample, batch_size, dataloader.dataset_object.text_embedding_dim, num_samples, augment
        )
        all_images += tf.squeeze(large_images).numpy().tolist()
        # TODO move below code out of for loop once we can generate images from stage 2
        # NOTE this uses a lot of memory so execute carefully
        all_images = np.array(all_images)
        is_avg, is_std = inception_score(experiment_name, all_images)
        print("Inception score: ", is_avg, "+/-", is_std)
        break
