import tensorflow as tf

class LearningRateDecay(object):

    def __init__(self, decay_factor: float, every_n: int = 1):
        """
        Arguments:
            decay_factor: float
                factor to multiply current learning rate by every_n epochs
            every_n: int
                decay learning rate every_n epochs
        """
        self.decay_factor = decay_factor
        self.every_n = every_n

    def __call__(self, model: tf.keras.Model, epoch_num: int):
        """
        Arguments:
            model: tf.keras.Model
                Model object which has optimizer parameter
            epoch_num: int
                Current epoch number
        """
        if not hasattr(model, 'optimizer'):
            raise Exception("model has no optimizer attribute, cannot run learning rate decay")

        if not hasattr(model.optimizer, 'lr'):
            raise Exception("model optimizer has no lr attribute, cannot run learning rate decay")

        if ((epoch_num + 1) % self.every_n) == 0:
            new_lr = model.optimizer.lr * self.decay_factor
            model.optimizer.lr.assign(new_lr)
