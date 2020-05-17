import numpy as np
import tensorflow as tf
from scipy.special import softmax
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model


def inception_score(experiment_name, images, n_split=10, eps=1E-16):
    """
    adapted from: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
    """
    path = f"results/{experiment_name}/inception/model"
    model = load_model(path)
    processed = np.asarray((images + 1) * 255. / 2, np.uint8)
    processed = preprocess_input(processed)
    yhat = model.predict(processed, batch_size=1)
    yhat = softmax(yhat, axis=1)
    scores = list()
    n_part = int(np.floor(images.shape[0] / n_split))
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = yhat[ix_start:ix_end]
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std
