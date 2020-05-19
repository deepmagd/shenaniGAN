import numpy as np
import tensorflow as tf

from shenanigan.metrics.InceptionScore import InceptionScore
from shenanigan.utils.data_helpers import tensors_from_sample


def evaluate(model, dataloader, experiment_name, num_samples, augment, noise_size):
    incep_score = InceptionScore(experiment_name)

    for _, sample in enumerate(dataloader.parsed_subset):
        batch_size = len(sample['text'].numpy())
        _, _, large_images, _, _ = tensors_from_sample(
            sample, batch_size, dataloader.dataset_object.text_embedding_dim, num_samples, augment
        )
        imgs = tf.squeeze(large_images).numpy()
        incep_score.predict_on_batch(imgs)
        # TODO move below code out of for loop once we can generate images from stage 2
    is_avg, is_std = incep_score.score()
    print("Inception score: ", is_avg, "+/-", is_std)
