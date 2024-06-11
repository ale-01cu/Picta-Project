import tensorflow_recommenders as tfrs
import tensorflow as tf
from typing import Dict, Text
from .tower.TowerModel import TowerModel
import typing as typ
import pandas as pd
from .data.DataPipelineBase import DataPipelineBase
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)



class MultitaskModel(tfrs.models.Model):
    def __init__(self, 
        rating_weight: float, 
        retrieval_weight: float, 
        towers_layers_sizes: typ.List[int],
        deep_layer_sizes: list[int],
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        train: tf.data.Dataset, 
        test: tf.data.Dataset,
        candidates: tf.data.Dataset,
        embedding_dimension: int = 32, 
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        candidates_batch: int = 128,
        k_candidates: int = 10,
    
    ) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()
        self.shuffle = shuffle
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.candidates_batch = candidates_batch
        self.k_candidates = k_candidates

        self.candidates = candidates
        self.cached_train = train.shuffle(self.shuffle)\
            .batch(self.train_batch).cache()
        self.cached_test = test.batch(self.test_batch).cache()
        self.cached_val = test.batch(self.test_batch).cache()

        self.query_model = TowerModel(
            layer_sizes=towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=features_data_q,
            embedding_dimension=embedding_dimension
        )
        self.candidate_model = TowerModel(
            layer_sizes=towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=features_data_c,
            embedding_dimension=embedding_dimension,
        )

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential()
        for size in deep_layer_sizes:
          self.rating_model.add(tf.keras.layers.Dense(size, activation="relu"))
        self.rating_model.add(tf.keras.layers.Dense(units=3, activation='sigmoid'))


        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
        )
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(
                candidates=self.candidates.batch(
                    self.candidates_batch).map(self.candidate_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight


    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embeddings = self.query_model(features)
        pub_embeddings = self.candidate_model(features)

        return (
            user_embeddings,
            pub_embeddings,
            self.rating_model(
                tf.concat([user_embeddings, pub_embeddings], axis=1)
            ),
        )


    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        catetories = features.pop("category")
        user_embeddings, pub_embeddings, categories_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=catetories,
            predictions=categories_predictions,
        )
        retrieval_loss = self.retrieval_task(
            user_embeddings, 
            pub_embeddings
        )

        # And combine them using the loss weights.
        return (
            self.rating_weight * rating_loss + 
            self.retrieval_weight * retrieval_loss
        )
    


if __name__ == '_main_':
    print('Cargando la data...')
    pubs_df = pd.read_csv('../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    views_df = pd.read_csv('../datasets/vistas_no_nulas.csv')
    positive_df = pd.read_csv('../datasets/positive_data.csv')
    features = ['usuario_id', 'id']

    pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))

    pipeline = DataPipelineBase(
        dataframe_path='../datasets/vistas_no_nulas.csv')
    pipeline.dataframe = pipeline.dataframe.drop(['id'], axis=1)
    
    df = pipeline.merge_data(
        df_to_merge=pubs_df, 
        left_on='publicacion_id',
        right_on='id',
        output_features=features
    )

    ds = pipeline.convert_to_tf_dataset(df)

    print('Construyendo vocabulario...')
    vocabularies = pipeline.build_vocabularies(
        features=features, ds=ds, batch=1_000)

    total, train_Length, val_length, test_length = pipeline.get_lengths(ds)

    train, val, test = pipeline.split_into_train_and_test(
        ds=ds,
        shuffle=4_000_000,
        train_length=train_Length,
        val_length=val_length,
        test_length=test_length,
        seed=42
    )

    model = MultitaskModel(
        rating_weight=1, 
        retrieval_weight=1, 
        towers_layers_sizes=[],
        deep_layer_sizes=[],
        vocabularies=vocabularies,
        features_data_q={
            'usuario_id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3 }    
        },
        features_data_c={ 
            'id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'nombre': { 'dtype': StringText.StringText, 'w': 0.2 },
            # 'descripcion': { 'dtype': StringText.StringText, 'w': 0.1 }
        },
        embedding_dimension=64, 
        train=train, 
        test=test, 
        val=val,
        shuffle=1_000_000, 
        train_batch=16_384, 
        test_batch=4096, 
        candidates=pubs_ds,
        candidates_batch=128, 
        k_candidates=100
    )
    

