import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense
from typing import Tuple

def build(classes: int, learning_rate: float, input_shape: Tuple[int, int, int]):
    base_model = InceptionV3(
        include_top=False,
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet'
    )
    base_model.trainable = True
    # set last inception block to trainable too (from mixed_9 onwards)
    for layer in base_model.layers[:280]:
        layer.trainable = False
    output = Dense(units=classes, activation=None)
    model = tf.keras.Sequential([
        base_model,
        output
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    return model
