import tensorflow as tf
from .data_pipeline import unique_users_ids, ratings_ds
from tensorflow.python.types.core import Tensor

class UserModel(tf.keras.Model):

    def __init__(self, embedding_dimension: int = 32) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension

        self.id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_users_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_users_ids) + 1, self.embedding_dimension),
        ])

        # self.timestamp_embedding = tf.keras.Sequential([
        #   tf.keras.layers.Discretization(timestamp_buckets.tolist()),
        #   tf.keras.layers.Embedding(len(timestamp_buckets) + 2, self.embedding_dimension)
        # ])

        # self.normalized_timestamp = tf.keras.layers.Normalization(axis=None)
        # self.normalized_timestamp.adapt(
        #     ratings_ds.map(lambda x: x["timestamp"]).batch(128))

    def call(self, inputs) -> Tensor:
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([
            self.id_embedding(inputs["user_id"]),
            # self.timestamp_embedding(inputs["timestamp"]),
            # tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
        ], axis=1)
  

if __name__ == "__main__":
    # --- Testeando el modelo ---
    user_model = UserModel()

    for row in ratings_ds.batch(1).take(1):
        print(f"Computed representations: {user_model(row)[0, :3]}")