import numpy as np
import tensorflow as tf

from shenanigan.metrics.InceptionScore import InceptionScore
from shenanigan.utils.data_helpers import tensors_from_sample


def evaluate(stage_1_generator, stage_2_generator, dataloader, experiment_name, num_samples, augment, noise_size):
    incep_score = InceptionScore(experiment_name)

    for _, sample in enumerate(dataloader.parsed_subset):
        batch_size = len(sample['text'].numpy())
        _, _, _, _, text_tensor = tensors_from_sample(
            sample, batch_size, dataloader.dataset_object.text_embedding_dim, num_samples, augment
        )
        noise_z = tf.random.normal((batch_size, noise_size))
        fake_images_small, _, _ = stage_1_generator([text_tensor, noise_z], training=False)
        fake_images_large, _, _ = stage_2_generator([fake_images_small, text_tensor], training=False)
        images = tf.squeeze(fake_images_large).numpy()
        incep_score.predict_on_batch(images)
    is_avg, is_std = incep_score.score(save=True)
    print("Inception score: ", is_avg, "+/-", is_std)
