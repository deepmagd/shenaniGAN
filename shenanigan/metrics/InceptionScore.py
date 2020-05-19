import os
import numpy as np
from scipy.special import softmax
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

class InceptionScore(object):
    def __init__(self, experiment_name: str, n_split: int = 10, eps: float = 1E-16):
        base_path = os.path.join('results', experiment_name)
        self.model_path = os.path.join(base_path, 'inception', 'model')
        self.save_path = os.path.join(base_path, 'inception_score.csv')
        self.n_split = n_split
        self.eps = eps
        self.model = self._load_model()
        self.predictions = []

    def _load_model(self):
        return load_model(self.model_path)

    def _save_scores(self, mean: float, std: float):
        with open(self.save_path, 'w+') as fd:
            write_str = f'{mean},{std}'
            fd.write(write_str)

    def predict_on_batch(self, images: np.ndarray):
        processed = np.asarray((images + 1) * 255. / 2, np.uint8)
        processed = preprocess_input(processed)
        yhat = self.model.predict(processed)
        yhat = softmax(yhat, axis=1)
        self.predictions += yhat.tolist()

    def score(self, save: bool = False):
        """
        adapted from: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
        """
        self.predictions = np.asarray(self.predictions)

        scores = []
        n_part = int(np.floor(len(self.predictions) / self.n_split))
        for i in range(self.n_split):
            ix_start, ix_end = i * n_part, i * n_part + n_part
            p_yx = self.predictions[ix_start:ix_end]
            p_y = np.expand_dims(p_yx.mean(axis=0), 0)
            kl_d = p_yx * (np.log(p_yx + self.eps) - np.log(p_y + self.eps))
            sum_kl_d = kl_d.sum(axis=1)
            avg_kl_d = np.mean(sum_kl_d)
            is_score = np.exp(avg_kl_d)
            scores.append(is_score)
        is_mean = np.mean(scores)
        is_std = np.std(scores)
        if save:
            self._save_scores(is_mean, is_std)
        return is_mean, is_std
