import tensorflow as tf
from shenanigan.utils import rmdir


class Checkpointer(object):
    def __init__(self, model: tf.keras.Model, save_dir: str, max_keep: int = None):
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0),
            discriminator=model.discriminator,
            generator=model.generator,
            g_optimizer=model.generator.optimizer,
            d_optimizer=model.discriminator.optimizer,
            loss=tf.Variable(1e06),  # some large number
        )
        self.checkpoint_dir = save_dir
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_dir, max_to_keep=max_keep
        )

    def restore(self, use_pretrained: bool = False, evaluate: bool = False):
        if use_pretrained:
            if evaluate:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial()
            else:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            if self.ckpt_manager.latest_checkpoint:
                print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            else:
                print("Initializing model from scratch")
        else:
            rmdir(self.checkpoint_dir)
            print("Initializing model from scratch")

    def save(self):
        return self.ckpt_manager.save(checkpoint_number=self.get_epoch_num())

    def get_epoch_num(self):
        return int(self.ckpt.step)

    def get_loss(self):
        return float(self.ckpt.loss)

    def increment_epoch(self):
        self.ckpt.step.assign_add(1)

    def update_loss(self, loss):
        self.ckpt.loss.assign(loss)
