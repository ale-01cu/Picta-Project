import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional
import pandas as pd
from recommender_engine.data.DataPipelineBase import DataPipelineBase
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)
import numpy as np

class FeaturesLayers(tf.keras.Model):

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_data: Dict[Text, object],
        embedding_dimension: int = 32,
        aditional_layers: Optional[list[object]] = None,
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocabularies = vocabularies
        self.features_data = features_data

        self.max_tokens = 100_000
        self.models = {}


        # self.title_weight = tf.Variable(0.3, trainable=True)
        # # self.title_text_weight = tf.Variable(1., trainable=True)
        # self.description_weight = tf.Variable(0.1, trainable=True)
        # self.category_weight = tf.Variable(0.2, trainable=True)

        for feature_name, feature_type in self.features_data.items():
            feature_data = vocabularies[feature_name]
            vocabulary = feature_data['vocabulary']

            if feature_type == CategoricalInteger.CategoricalInteger:
                model = tf.keras.Sequential([
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(
                        vocabulary) + 1, self.embedding_dimension),
                ])
            
            elif feature_type == CategoricalString.CategoricalString:
                model = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension),
                ])

            elif feature_type == CategoricalContinuous.CategoricalContinuous:
                vocabulary = tf.data.Dataset.from_tensor_slices({'timestamp': vocabulary})
                max_timestamp = vocabulary.map(lambda x: x["timestamp"]).reduce(
                    tf.cast(0, tf.int64), tf.maximum).numpy().max()
                min_timestamp = vocabulary.map(lambda x: x["timestamp"]).reduce(
                    np.int64(1e9), tf.minimum).numpy().min()

                timestamp_buckets = np.linspace(
                    min_timestamp, max_timestamp, num=1000)


                model = tf.keras.Sequential([
                    tf.keras.layers.Discretization(timestamp_buckets.tolist()),
                    tf.keras.layers.Embedding(len(timestamp_buckets) + 2, self.embedding_dimension)
                ])

                self.normalized_timestamp = tf.keras.layers.Normalization(
                    axis=None
                )

                # tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))



            elif feature_type == StringText.StringText:
                vectorization_layer = tf.keras.layers.TextVectorization(
                    max_tokens=self.max_tokens)
                vectorization_layer.adapt(vocabulary)

                model = tf.keras.Sequential([
                    vectorization_layer,
                    tf.keras.layers.Embedding(self.max_tokens, 
                        self.embedding_dimension, mask_zero=True),
                    tf.keras.layers.GlobalAveragePooling1D(),
                ])


            if aditional_layers:
                for layer in aditional_layers:
                    model.add(layer)
            self.models[feature_name] = model


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
            for feature in self.features_data
        ], axis=1)




if __name__ == "__main__":
    likes_df_path = 'I:/UCI/tesis/Picta-Project/datasets/likes_con_timestamp_100K.csv'
    pubs_df = pd.read_csv('I:/UCI/tesis/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    features = ['id', 'nombre', 'descripcion', 'timestamp']

    pipeline = DataPipelineBase(dataframe_path=likes_df_path)

    df = pipeline.merge_data(
        df_to_merge=pubs_df, 
        left_on='publication_id',
        right_on='id',
        output_features=features
    )

    df['nombre'] = df['nombre'].astype(str)
    df['descripcion'] = df['descripcion'].astype(str)
    
    ds = pipeline.convert_to_tf_dataset(df)

    vocabularies = pipeline.build_vocabularies(
        features=features, 
        ds=ds, 
        batch=1_000
    )
    
    total, train_Length, val_length, test_length = pipeline.get_lengths(ds)

    print(vocabularies['timestamp']['vocabulary'])

    # --- Testeando el modelo ---
    features_layers = FeaturesLayers(
        vocabularies=vocabularies, 
        features_data={
            'id': CategoricalInteger.CategoricalInteger,
            'nombre': CategoricalString.CategoricalString,
            'descripcion': StringText.StringText,
            'timestamp': CategoricalContinuous.CategoricalContinuous
        },
        embedding_dimension=64
    )

    for row in ds.batch(1).take(1):
        print(f"Computed representations: {features_layers(row)[0, :3]}")

    features_layers.summary()