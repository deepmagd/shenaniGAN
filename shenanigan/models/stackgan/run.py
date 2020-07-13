import os

import tensorflow as tf
from typing import Tuple

from shenanigan.callbacks import LearningRateDecay
from shenanigan.utils import extract_epoch_num
from shenanigan.utils.logger import LogPlotter
from shenanigan.visualise import compare_generated_to_real
from shenanigan.utils.model_helpers import Checkpointer

from . import StackGAN1, StackGAN2
from .evaluate import evaluate as eval_fxn
from .utils import get_trainer


def load_model(
    settings,
    image_dims: Tuple[int, int],
    results_dir: str,
    stage: int,
    epoch_num: int = -1,
) -> StackGAN1:
    model = StackGAN1(
        img_size=image_dims,
        lr_g=settings["stage1"]["generator"]["learning_rate"],
        lr_d=settings["stage1"]["discriminator"]["learning_rate"],
        conditional_emb_size=settings["stage1"]["conditional_emb_size"],
        w_init=tf.random_normal_initializer(stddev=0.02),
        bn_init=tf.random_normal_initializer(1.0, 0.02),
    )
    if stage == 2:
        results_dir = results_dir.replace("stage-2", "stage-1")

    if epoch_num == -1:
        # Find last checkpoint
        epoch_num = extract_epoch_num(results_dir)

    pretrained_dir = os.path.join(results_dir, f"model_{epoch_num}")
    model.generator = tf.saved_model.load(
        os.path.join(pretrained_dir, "generator", "generator")
    )
    model.discriminator = tf.saved_model.load(
        os.path.join(pretrained_dir, "discriminator", "discriminator")
    )
    return model


def run(
    train_loader: object,
    val_loader: object,
    small_image_dims,
    results_dir: str,
    settings,
    experiment_name: str,
    stage: int,
    use_pretrained: bool = False,
    visualise: bool = False,
    evaluate: bool = False,
):
    lr_decay = LearningRateDecay(
        decay_factor=settings["callbacks"]["learning_rate_decay"]["decay_factor"],
        every_n=settings["callbacks"]["learning_rate_decay"]["every_n"],
    )

    # use best when doing inference
    checkpoint_dir = os.path.join(results_dir, "ckpts_every")

    if stage == 1 and evaluate and visualise:
        model = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings["stage1"]["generator"]["learning_rate"],
            lr_d=settings["stage1"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage1"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer = Checkpointer(
            model=model,
            save_dir=checkpoint_dir.replace("stage-2", "stage-1"),
            max_keep=None,
        )
        checkpointer.restore(use_pretrained=True, evaluate=True)
        compare_generated_to_real(
            dataloader=train_loader,
            num_images=settings["visualisation"]["images_to_generate"],
            noise_size=settings["stage1"]["noise_size"],
            model=model,
            save_location=os.path.join(results_dir, "viz"),
            img_size="small",
        )

    elif stage == 2 and evaluate and visualise:

        model_stage1 = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings["stage1"]["generator"]["learning_rate"],
            lr_d=settings["stage1"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage1"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer_stage1 = Checkpointer(
            model=model_stage1,
            save_dir=checkpoint_dir.replace("stage-2", "stage-1"),
            max_keep=None,
        )
        checkpointer_stage1.restore(use_pretrained=True, evaluate=True)

        model_stage2 = StackGAN2(
            img_size=small_image_dims,
            lr_g=settings["stage2"]["generator"]["learning_rate"],
            lr_d=settings["stage2"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage2"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer_stage2 = Checkpointer(
            model=model_stage2, save_dir=checkpoint_dir, max_keep=None
        )
        checkpointer_stage2.restore(use_pretrained=True, evaluate=True)
        compare_generated_to_real(
            dataloader=train_loader,
            num_images=settings["visualisation"]["images_to_generate"],
            noise_size=settings["stage1"]["noise_size"],
            model=model_stage1,
            save_location=os.path.join(results_dir, "viz"),
            img_size="large",
            subsequent_model=model_stage2,
        )

    elif stage == 1 and evaluate:
        raise NotImplementedError("Evaluation for stage 1 is not implemented")

    elif stage == 2 and evaluate:
        model_stage1 = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings["stage1"]["generator"]["learning_rate"],
            lr_d=settings["stage1"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage1"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer_stage1 = Checkpointer(
            model=model_stage1,
            save_dir=checkpoint_dir.replace("stage-2", "stage-1"),
            max_keep=None,
        )
        checkpointer_stage1.restore(use_pretrained=True, evaluate=True)

        model_stage2 = StackGAN2(
            img_size=small_image_dims,
            lr_g=settings["stage2"]["generator"]["learning_rate"],
            lr_d=settings["stage2"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage2"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer_stage2 = Checkpointer(
            model=model_stage2, save_dir=checkpoint_dir, max_keep=None
        )
        checkpointer_stage2.restore(use_pretrained=True, evaluate=True)

        eval_fxn(
            stage_1_generator=model_stage1.generator,
            stage_2_generator=model_stage2.generator,
            dataloader=val_loader,
            experiment_name=experiment_name,
            num_samples=settings["stage2"]["num_samples"],
            augment=True,
            noise_size=settings["stage1"]["noise_size"],
        )

    elif stage == 1:
        model = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings["stage1"]["generator"]["learning_rate"],
            lr_d=settings["stage1"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage1"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )

        trainer_class = get_trainer(stage)
        trainer = trainer_class(
            model=model,
            batch_size=settings["common"]["batch_size"],
            save_location=results_dir,
            save_every=settings["stage1"]["save_every_n_epochs"],
            save_best_after=settings["stage1"]["save_best_after_n_epochs"],
            callbacks=[lr_decay],
            use_pretrained=use_pretrained,
            num_samples=settings["stage1"]["num_samples"],
            noise_size=settings["stage1"]["noise_size"],
            augment=settings["stage1"]["augment"],
        )
        trainer(train_loader, val_loader, num_epochs=settings["stage1"]["num_epochs"])
        plotter = LogPlotter(results_dir)
        plotter.learning_curve()

    elif stage == 2:
        model_stage1 = StackGAN1(
            img_size=small_image_dims,
            lr_g=settings["stage1"]["generator"]["learning_rate"],
            lr_d=settings["stage1"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage1"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )
        checkpointer = Checkpointer(
            model=model_stage1,
            save_dir=checkpoint_dir.replace("stage-2", "stage-1"),
            max_keep=None,
        )
        checkpointer.restore(use_pretrained=True, evaluate=True)

        model_stage2 = StackGAN2(
            img_size=small_image_dims,
            lr_g=settings["stage2"]["generator"]["learning_rate"],
            lr_d=settings["stage2"]["discriminator"]["learning_rate"],
            conditional_emb_size=settings["stage2"]["conditional_emb_size"],
            w_init=tf.random_normal_initializer(stddev=0.02),
            bn_init=tf.random_normal_initializer(1.0, 0.02),
        )

        trainer_class = get_trainer(stage)
        trainer = trainer_class(
            model=model_stage2,
            batch_size=settings["common"]["batch_size"],
            save_location=results_dir,
            save_every=settings["stage2"]["save_every_n_epochs"],
            save_best_after=settings["stage2"]["save_best_after_n_epochs"],
            callbacks=[lr_decay],
            use_pretrained=use_pretrained,
            num_samples=settings["stage2"]["num_samples"],
            noise_size=settings["stage1"]["noise_size"],
            augment=settings["stage2"]["augment"],
            stage_1_generator=model_stage1.generator,
        )

        trainer(train_loader, val_loader, num_epochs=settings[f"stage2"]["num_epochs"])
        plotter = LogPlotter(results_dir)
        plotter.learning_curve()
