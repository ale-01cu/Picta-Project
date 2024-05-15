import tensorflow as tf
from .FeaturesLayers import FeaturesLayers
from tensorflow.python.types.core import Tensor
from typing import Dict, Text, Optional
from recommender_engine.data.DataPipelineBase import DataPipelineBase
import pandas as pd
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)

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
        self.dense_layers.add(tf.keras.layers.Dropout(0.2))

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



if __name__ == "__main__":
    ratings_df_path = 'I:/UCI/tesis/Picta-Project/datasets/publicaciones_ratings_con_timestamp.csv'
    pubs_df = pd.read_csv('I:/UCI/tesis/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    features = ['id', 'nombre', 'descripcion', 'timestamp']
    
    pipeline = DataPipelineBase(dataframe_path=ratings_df_path)

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
        features=features, ds=ds, batch=1_000)
    

    # --- Testeando el modelo ---
    pub_model = TowerModel(
        vocabularies=vocabularies, 
        features_data={
            'id': {'dtype': CategoricalInteger.CategoricalInteger, 'w': 0.1},
            'nombre': {'dtype': CategoricalString.CategoricalString, 'w': 0.2},
            'descripcion': {'dtype': StringText.StringText, 'w': 0.1},
            'timestamp': {'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3}
        },
        embedding_dimension=64,
        max_tokens=10_000,
        layer_sizes=[128, 128, 128]
    )

    # for row in ratings_ds.batch(1).take(1):
    #   print(f"Computed representations: {pub_model(row)[0, :3]}")

    a = ds.batch(128).map(lambda x: pub_model(x))

    for i in a.take(1).as_numpy_iterator():
        print(i)