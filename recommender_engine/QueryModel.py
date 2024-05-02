import tensorflow as tf
from .UserModel import UserModel
from tensorflow.python.types.core import Tensor
from typing import Dict, Text
import pandas as pd
from .data.DataPipelineBase import DataPipelineBase

class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_names: list[str],
        layer_sizes: list[int], 
        embedding_dimension: int
    ) -> None:
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel(
            vocabularies=vocabularies,
            features_names=features_names,
            embedding_dimension=embedding_dimension
        )

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))


    def call(self, inputs) -> Tensor:
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
  

if __name__ == "__main__":
    ratings_df_path = 'C:/Users/Ale/Desktop/Picta-Project/datasets/publicaciones_ratings_con_timestamp_medium.csv'
    pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    
    pipeline = DataPipelineBase(dataframe_path=ratings_df_path)
    df = pipeline.merge_data(
        df_to_merge=pubs_df, 
        left_on='publication_id',
        right_on='id',
        output_features=['user_id', 'publication_id']
    )

    ds = pipeline.convert_to_tf_dataset(df)

    vocabularies = pipeline.build_vocabularies(
        features=['user_id', 'publication_id'], ds=ds, batch=1_000)
    
    total, train_Length, test_length = pipeline.get_lengths(ds)

    print(pipeline)


    # --- Testeando el modelo ---
    q_model = QueryModel(
        vocabularies=vocabularies, 
        features_names=['user_id'],
        layer_sizes=[64], 
        embedding_dimension=64
    )
    # for row in ratings_ds.batch(1).take(1):
    #   print(f"Computed representations: {q_model(row)[0, :3]}")

    a = ds.batch(128).map(q_model)
    for i in a.take(1).as_numpy_iterator():
        print(i)