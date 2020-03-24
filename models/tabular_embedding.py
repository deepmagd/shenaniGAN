from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class TabularEmbedding(Model):
    def __init__(self, output_embedding_dim):
        """ During initialisation over here we need to set out all the possible
            encodings for each features. This dictates the tabular_to_vector method
        """
        self.encoding_map = some_function()
        self.fc = Dense(units=output_embedding_dim, activation='sigmoid')

    def __call__(self, x):
        """ Recieve a tabular row, encode it into a vector,
            then pass it through a simple network to get a dense vector.
        """
        initial_vector = self.tabular_to_vector(x)
        return self.fc(initial_vector)

    def tabular_to_vector(self, x):
        """ Recieve a tabular row, deterministically encode it into a vector """
        pass