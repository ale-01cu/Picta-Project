import tensorflow as tf
from engine.models.tower.FeaturesLayers import FeaturesLayers
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
        regularization_l2: float = 0.1,
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

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(
                layer_size))
            
        l2_regularizer = tf.keras.regularizers.L2(l2=regularization_l2)

        self.dense_layers.add(tf.keras.layers.Dense(
            64, 
            activation='relu', 
            kernel_regularizer=l2_regularizer, 
            bias_regularizer=l2_regularizer
        ))
            


    def call(self, inputs) -> Tensor:
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'vocabularies': self.embedding_model.get_config()['vocabularies'],
    #         'features_data': self.embedding_model.get_config()['features_data'],
    #         'embedding_dimension': self.embedding_model.get_config()['embedding_dimension'],
    #         'max_tokens': self.embedding_model.get_config()['max_tokens'],
    #         'layer_sizes': [layer.units for layer in self.dense_layers.layers],
    #         'regularization_l2': self.dense_layers.layers[-1].kernel_regularizer.l2,
    #         'aditional_layers': self.embedding_model.get_config()['aditional_layers'],
    #     })
    #     return config
    
    # @classmethod
    # def from_config(cls, config):
    #     # Deserializar las capas adicionales si existen
    #     if "aditional_layers" in config:
    #         config["aditional_layers"] = [tf.keras.layers.deserialize(layer_config) 
    #                                       for layer_config in config["aditional_layers"]]
        
    #     return cls(**config)