import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional, Union, List
import pandas as pd
from recommender_engine.data.DataPipelineBase import DataPipelineBase
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)
import numpy as np

class FeaturesLayers(tf.keras.Model):
    embedding_dimension: int
    vocabularies: Dict[Text, Dict[Text, tf.Tensor]]
    features_data = Dict[Text, Dict[Text, object]]
    features_weights = Dict[Text, float]
    max_tokens: int
    models: Dict[Text, tf.keras.Sequential]
    extra_layers: Dict[Text, tf.keras.layers.Layer]

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_data: Dict[Text, Dict[Text, object]],
        embedding_dimension: int = 32,
        max_tokens: int = 10_000,
        aditional_layers: Optional[list[object]] = None,
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.vocabularies = vocabularies
        self.features_data = features_data
        self.features_weights = {}

        self.max_tokens = max_tokens
        self.models = {}
        self.extra_layers = {}


        for feature_name, feature_data in self.features_data.items():
            feature_type = feature_data['dtype']
            feature_weight = feature_data['w']

            self.features_weights[feature_name] = feature_weight

            feature_vocabulary = vocabularies[feature_name]
            vocabulary = feature_vocabulary['vocabulary']

            print(f'Building {feature_name} feature...')

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
                vocabulary = tf.data.Dataset.from_tensor_slices({str(feature_name): vocabulary})
                vocabulary = vocabulary.map(lambda x: x[str(feature_name)])


                max_timestamp = vocabulary.reduce(
                    tf.cast(0, tf.int64), tf.maximum).numpy().max()
                min_timestamp = vocabulary.reduce(
                    np.int64(1e9), tf.minimum).numpy().min()

                timestamp_buckets = np.linspace(
                    min_timestamp, max_timestamp, num=1000)


                model = tf.keras.Sequential([
                    tf.keras.layers.Discretization(timestamp_buckets.tolist()),
                    tf.keras.layers.Embedding(len(timestamp_buckets) + 2, self.embedding_dimension)
                ])

                normalized_timestamp = tf.keras.layers.Normalization(
                    axis=None
                )

                normalized_timestamp.adapt(vocabulary.batch(128))
                self.extra_layers[feature_name] = normalized_timestamp


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
        features_embeddings = [
            self.models[feature](inputs[feature]) * self.features_weights[feature]
            for feature in self.features_data
        ]

        if self.extra_layers.values():
            extra_layers = tf.concat([
                tf.reshape(model(inputs[feature]), (-1, 1)) * self.features_weights[feature]
                for feature, model in self.extra_layers.items()
            ], axis=1)

            features_embeddings.append(extra_layers) 
        
        return tf.concat(features_embeddings, axis=1)





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
            'id': {'dtype': CategoricalInteger.CategoricalInteger, 'w': 0.1},
            'nombre': {'dtype': CategoricalString.CategoricalString, 'w': 0.2},
            'descripcion': {'dtype': StringText.StringText, 'w': 0.1},
            'timestamp': {'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3}
        },
        embedding_dimension=64,
        max_tokens=10_000
    )


    for row in ds.batch(1).take(1):
        print(f"Computed representations: {features_layers(row)}")

    features_layers.summary()


    print('Pesos: ', features_layers.features_weights)