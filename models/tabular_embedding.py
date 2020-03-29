from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class Perceptron(Model):
    def __init__(self, output_embedding_dim):
        """ An incredibly simple MLP """
        self.fc = Dense(units=output_embedding_dim, activation='sigmoid')

    def __call__(self, x):
        """ Forward pass """
        return self.fc(x)
