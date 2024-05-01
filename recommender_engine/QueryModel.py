import tensorflow as tf
from .UserModel import UserModel
from .data_pipeline import ratings_ds
from tensorflow.python.types.core import Tensor

class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes: list[int], embedding_dimension: int) -> None:
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel(embedding_dimension=embedding_dimension)

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
    q_model = QueryModel([64], 64)
    # for row in ratings_ds.batch(1).take(1):
    #   print(f"Computed representations: {q_model(row)[0, :3]}")

    a = ratings_ds.batch(128).map(lambda x: (x['user_id'], q_model(x)))
    for i in a.take(1).as_numpy_iterator():
        print(i)