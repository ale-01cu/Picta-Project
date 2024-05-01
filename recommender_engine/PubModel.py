import tensorflow as tf
from .data_pipeline import unique_pubs_names, ratings_ds
from tensorflow.python.types.core import Tensor

class PubModel(tf.keras.Model):

    def __init__(self, embedding_dimension: int = 32) -> None:
        super().__init__()

        self.max_tokens = 10_000
        self.embedding_dimension = embedding_dimension

        # self.title_weight = tf.Variable(0.3, trainable=True)
        # # self.title_text_weight = tf.Variable(1., trainable=True)
        # self.description_weight = tf.Variable(0.1, trainable=True)
        # self.category_weight = tf.Variable(0.2, trainable=True)

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_pubs_names, mask_token=None),
            tf.keras.layers.Embedding(len(unique_pubs_names) + 1, self.embedding_dimension)
        ])

        # self.title_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens)
        # self.title_vectorization_layer.adapt(pubs_names_ds.batch(128))

        # self.title_embedding = tf.keras.Sequential([
        #   self.title_vectorization_layer,
        #   tf.keras.layers.Embedding(self.max_tokens, self.embedding_dimension, mask_zero=True),
        #   tf.keras.layers.GlobalAveragePooling1D(),
        # ])

        # self.description_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=self.max_tokens)
        # self.description_vectorization_layer.adapt(pubs_descriptions_ds.batch(128))

        # self.description_embedding = tf.keras.Sequential([
        #   self.description_vectorization_layer,
        #   tf.keras.layers.Embedding(self.max_tokens, self.embedding_dimension, mask_zero=True),
        #   tf.keras.layers.GlobalAveragePooling1D(),
        # ])

        # self.category_embedding = tf.keras.Sequential([
        #   tf.keras.layers.StringLookup(vocabulary=unique_pubs_categories, mask_token=None),
        #   tf.keras.layers.Embedding(len(unique_pubs_categories) + 1, self.embedding_dimension)
        # ])


    def call(self, inputs) -> Tensor:
        return tf.concat([
            # self.id_embedding(inputs['publication_id']),
            self.title_embedding(inputs["nombre"]),
            # self.title_embedding(inputs["nombre"]) * self.title_weight,
            # self.category_embedding(inputs["categoria"]) * self.category_weight,
            # self.description_embedding(inputs["descripcion"]) * self.description_weight,
        ], axis=1)


if __name__ == "__main__":
    # --- Testeando el modelo
    pub_model = PubModel()

    for row in ratings_ds.batch(1).take(1):
        print(f"Computed representations: {pub_model(row)[0, :3]}")