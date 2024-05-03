import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional
import pandas as pd
from .data.DataPipelineBase import DataPipelineBase

class PubModel(tf.keras.Model):

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_names: list[str],
        embedding_dimension: int = 32,
        aditional_layers: Optional[list[object]] = None,
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocabularies = vocabularies
        self.features_names = features_names

        self.max_tokens = 10_000
        self.models = {}


        # self.title_weight = tf.Variable(0.3, trainable=True)
        # # self.title_text_weight = tf.Variable(1., trainable=True)
        # self.description_weight = tf.Variable(0.1, trainable=True)
        # self.category_weight = tf.Variable(0.2, trainable=True)

        for feature_name in self.features_names:
            feature_data = vocabularies[feature_name]
            vocabulary = feature_data['vocabulary']

            if feature_data['dtype'] == tf.int64 or feature_data['dtype'] == tf.int32:
                model = tf.keras.Sequential([
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(
                        vocabulary) + 1, self.embedding_dimension),
                ])
            
            elif feature_data['dtype'] == tf.string:
                model = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension),
                ])

            if aditional_layers:
                for layer in aditional_layers:
                    model.add(layer)
            self.models[feature_name] = model

        # self.title_embedding = tf.keras.Sequential([
        #     tf.keras.layers.StringLookup(vocabulary=unique_pubs_names, mask_token=None),
        #     tf.keras.layers.Embedding(len(unique_pubs_names) + 1, self.embedding_dimension)
        # ])

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
        # return tf.concat([
        #     # self.id_embedding(inputs['publication_id']),
        #     self.title_embedding(inputs["nombre"]),
        #     # self.title_embedding(inputs["nombre"]) * self.title_weight,
        #     # self.category_embedding(inputs["categoria"]) * self.category_weight,
        #     # self.description_embedding(inputs["descripcion"]) * self.description_weight,
        # ], axis=1)

        return tf.concat([
            self.models[feature](inputs[feature]) 
            for feature in self.features_names
        ], axis=1)


if __name__ == "__main__":
    ratings_df_path = 'C:/Users/Ale/Desktop/Picta-Project/datasets/publicaciones_ratings_con_timestamp_medium.csv'
    pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v4.csv')
    
    pipeline = DataPipelineBase(dataframe_path=ratings_df_path)

    df = pipeline.merge_data(
        df_to_merge=pubs_df, 
        left_on='publication_id',
        right_on='id',
        output_features=['user_id', 'publication_id', 'nombre']
    )

    df['nombre'] = df['nombre'].astype(str)
    
    ds = pipeline.convert_to_tf_dataset(df)

    vocabularies = pipeline.build_vocabularies(
        features=['user_id', 'publication_id', 'nombre'], 
        ds=ds, 
        batch=1_000
    )
    
    total, train_Length, test_length = pipeline.get_lengths(ds)


    # --- Testeando el modelo ---
    pub_model = PubModel(
        vocabularies=vocabularies, 
        features_names=['nombre'],
        embedding_dimension=64
    )

    for row in ds.batch(1).take(1):
        print(f"Computed representations: {pub_model(row)[0, :3]}")