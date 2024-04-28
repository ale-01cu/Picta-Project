import tensorflow as tf
from .UserModel import UserModel
from .data_pipeline import ratings_ds

class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
            layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))


    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
  

if __name__ == "__main__":
    # --- Testeando el modelo ---
    q_model = QueryModel([64])
    # for row in ratings_ds.batch(1).take(1):
    #   print(f"Computed representations: {q_model(row)[0, :3]}")

    a = ratings_ds.batch(128).map(lambda x: (x['user_id'], q_model(x)))
    for i in a.take(1).as_numpy_iterator():
        print(i)