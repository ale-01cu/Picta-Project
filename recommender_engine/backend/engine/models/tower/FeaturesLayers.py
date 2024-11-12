import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional
from engine.data.FeaturesTypes import (
    StringText,
    CategoricalContinuous,
    CategoricalString,
    CategoricalInteger
)
import numpy as np

class FeaturesLayers(tf.keras.Model):
    embedding_dimension: int
    vocabularies: Dict[Text, Dict[Text, tf.Tensor]]
    features_data = Dict[Text, Dict[Text, object]]
    features_weights = Dict[Text, float]
    max_tokens: int
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
        self.aditional_layers = aditional_layers
        self.extra_layers = {}
        self.model = None


        for feature_name, feature_data in self.features_data.items():
            feature_type = feature_data['dtype']
            feature_weight = feature_data['w']

            self.features_weights[feature_name] = feature_weight

            feature_vocabulary = vocabularies[feature_name]
            vocabulary = feature_vocabulary['vocabulary']

            print(f'Building {feature_name} feature...')

            if feature_type == CategoricalInteger:
                # normalization_layer = tf.keras.layers.Normalization(axis=None)
                # normalization_layer.adapt(vocabulary)

                self.model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(1,), name = feature_name + 'input', dtype = tf.int32),
                    #normalization_layer,
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(len(
                        vocabulary) + 1, self.embedding_dimension),
                    tf.keras.layers.Flatten(name='Flatten_' + feature_name)

                ])

            elif feature_type == CategoricalString:
                self.model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(1,), name = feature_name + '_input', dtype = tf.string),
                    tf.keras.layers.StringLookup(
                        vocabulary=vocabulary, mask_token=None),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1, self.embedding_dimension),
                    tf.keras.layers.Flatten(name='Flatten_' + feature_name)
                ])

            elif feature_type == CategoricalContinuous:
                vocabulary = tf.data.Dataset.from_tensor_slices({str(feature_name): vocabulary})
                vocabulary = vocabulary.map(lambda x: x[str(feature_name)])


                max_timestamp = vocabulary.reduce(
                    tf.cast(0, tf.int64), tf.maximum).numpy().max()
                min_timestamp = vocabulary.reduce(
                    np.int64(1e9), tf.minimum).numpy().min()

                timestamp_buckets = np.linspace(
                    min_timestamp, max_timestamp, num=1000)


                self.model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(1,), name = feature_name + '_input', dtype = tf.int64),
                    tf.keras.layers.Discretization(timestamp_buckets.tolist()),
                    tf.keras.layers.Embedding(len(timestamp_buckets) + 2, self.embedding_dimension),
                    tf.keras.layers.Flatten(name='Flatten_' + feature_name)
                ])

                normalized_timestamp = tf.keras.layers.Normalization(
                    axis=None
                )

                normalized_timestamp.adapt(vocabulary.batch(128))
                self.extra_layers[feature_name] = normalized_timestamp


            elif feature_type == StringText:
                vectorization_layer = tf.keras.layers.TextVectorization(
                    max_tokens=self.max_tokens)
                vectorization_layer.adapt(vocabulary)

                self.model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(
                        input_shape=(1,),
                        name = feature_name + '_input',
                        dtype = tf.string
                    ),
                    vectorization_layer,
                    tf.keras.layers.Embedding(self.max_tokens,
                        self.embedding_dimension, mask_zero=True),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    tf.keras.layers.Flatten(name='Flatten_' + feature_name)

                ])


            if self.aditional_layers:
                for layer in self.aditional_layers:
                    self.model.add(layer)
            self.models[feature_name] = self.model

            self.model = None


    def call(self, inputs) -> Tensor:
        features_embeddings = [
            self.models[feature](inputs[feature]) * self.features_weights[feature]
            for feature in self.features_data
            if feature in inputs.keys()
        ]

        if self.extra_layers.values():
            extra_layers = tf.concat([
                tf.reshape(model(inputs[feature]), (-1, 1)) * self.features_weights[feature]
                for feature, model in self.extra_layers.items()
                if feature in self.extra_layers.keys()
            ], axis=1)

            features_embeddings.append(extra_layers)

        return tf.concat(features_embeddings, axis=1)

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'embedding_dimension': self.embedding_dimension,
    #         'vocabularies': self.vocabularies,
    #         'features_data': self.features_data,
    #         'max_tokens': self.max_tokens,
    #         'extra_layers': self.extra_layers,
    #         'models': {name: model.get_config() for name, model in self.models.items()},
    #         'aditional_layers': [layer.get_config() for layer in self.aditional_layers ]
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     # Deserializar las configuraciones de los modelos
    #     config["models"] = {name: tf.keras.models.model_from_config(model_config)
    #                         for name, model_config in config["models"].items()}

    #     # Deserializar las capas adicionales si existen
    #     if "aditional_layers" in config:
    #         config["aditional_layers"] = [tf.keras.layers.deserialize(layer_config)
    #                                       for layer_config in config["aditional_layers"]]

    #     return cls(**config)