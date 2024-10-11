import tensorflow as tf
from engine.tower.FeaturesLayers import FeaturesLayers
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional

class TowerModel(tf.keras.Model):
    """Model for encoding movies."""
    embedding_model: tf.keras.Model
    dense_layers: tf.keras.Sequential

    def __init__(self, 
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_data: Dict[Text, Dict[Text, object]],
        embedding_dimension: int,
        layer_sizes: list[int], 
        max_tokens: int = 10_000,
        aditional_layers: Optional[list[object]] = None,
    ) -> None:
        super().__init__()

        self.embedding_model = FeaturesLayers(
            vocabularies=vocabularies,
            features_data=features_data,
            embedding_dimension=embedding_dimension,
            aditional_layers=aditional_layers,
            max_tokens=max_tokens
        )

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(tf.keras.layers.Dropout(0.5))

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size))
            


    def call(self, inputs) -> Tensor:
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
