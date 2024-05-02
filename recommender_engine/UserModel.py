import tensorflow as tf
from .data.DataPipelineBase import DataPipelineBase
from tensorflow.python.types.core import Tensor
from typing import Dict, Text
import pandas as pd

class UserModel(tf.keras.Model):

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_names: list[str],
        embedding_dimension: int = 32
    ) -> None:
        
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocabularies = vocabularies
        self.features_names = features_names

        self.models = {}

        for feature_name in self.features_names:
            feature_data = vocabularies[feature_name]
            vocabulary = feature_data['vocabulary']

            if feature_data['dtype'] == tf.int64 or feature_data['dtype'] == tf.int32:
                model = tf.keras.Sequential([
                    tf.keras.layers.IntegerLookup(vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1, self.embedding_dimension),
                ])

            elif feature_data['dtype'] == tf.string:
                model = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(vocabulary) + 1, self.embedding_dimension),
                ])

            self.models[feature_name] = model

        # self.id_embedding = tf.keras.Sequential([
        #     tf.keras.layers.StringLookup(vocabulary=unique_users_ids, mask_token=None),
        #     tf.keras.layers.Embedding(len(unique_users_ids) + 1, self.embedding_dimension),
        # ])

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
        # return tf.concat([
        #     self.id_embedding(inputs["user_id"]),
        #     # self.timestamp_embedding(inputs["timestamp"]),
        #     # tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
        # ], axis=1)
    
        return tf.concat([
            self.models[feature](inputs[feature]) 
            for feature in self.features_names
        ], axis=1)
  

if __name__ == "__main__":
    ratings_df_path = 'C:/Users/Ale/Desktop/Picta-Project/datasets/publicaciones_ratings_con_timestamp_medium.csv'
    pubs_df_path = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    
    pipeline = DataPipelineBase(dataframe_path=ratings_df_path)
    df = pipeline.merge_data(
        df_to_merge=pubs_df_path, 
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
    user_model = UserModel(vocabularies=vocabularies, features_names=['user_id'])

    for row in ds.batch(1).take(1):
        print(f"Computed representations: {user_model(row)[0, :3]}")