import tensorflow as tf
from .PubModel import PubModel
from .data_pipeline import pubs_ds
from tensorflow.python.types.core import Tensor

class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes: list[int], embedding_dimension: int) -> None:
        super().__init__()

        self.embedding_model = PubModel(embedding_dimension=embedding_dimension)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))


    def call(self, inputs) -> Tensor:
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


if __name__ == "__main__":
    # --- Testeando el modelo ---
    pub_model = CandidateModel([64])

    # for row in ratings_ds.batch(1).take(1):
    #   print(f"Computed representations: {pub_model(row)[0, :3]}")

    a = pubs_ds.batch(128).map(lambda x: (x['nombre'], pub_model(x)))

    for i in a.take(1).as_numpy_iterator():
        print(i)