import tensorflow as tf
import tensorflow_recommenders as tfrs
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from typing import Dict, Text
from .data_pipeline import pubs_df
import numpy as np

class RankingModel(tfrs.models.Model):
    """
    
    Los datos para entrenar este modelo son calificaciones
    que ha dado un usuario a un contenido
    
    ejemplo:

    El usuario x dio una calificacion z a la publicacion k.
    
    """

    def __init__(self, 
        layer_sizes: list[int], 
        deep_layer_sizes: list[int],
        train: tf.data.Dataset,
        test: tf.data.Dataset,
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        embedding_dimension: int = 32,
    ) -> None:
        
        super().__init__()

        self.cached_train = train.shuffle(shuffle).batch(train_batch).cache()
        self.cached_test = test.batch(test_batch).cache()


        self.query_model = QueryModel(
            layer_sizes=layer_sizes, 
            embedding_dimension=embedding_dimension
        )
        self.candidate_model = CandidateModel(
            layer_sizes=layer_sizes, 
            embedding_dimension=embedding_dimension
        )

        # Compute predictions.
        # self.rating_model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(256, activation="relu"),
        #     tf.keras.layers.Dense(128, activation="relu"),
        #     tf.keras.layers.Dense(1),
        # ])

        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(size, activation="relu") 
            for size in deep_layer_sizes
        ])
        self.rating_model.add(tf.keras.layers.Dense(1))


        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )


    def call(self, inputs: dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embedding = self.query_model(inputs)
        pub_embedding = self.candidate_model(inputs)

        return self.rating_model(tf.concat(
            [user_embedding, pub_embedding], axis=1))


    def compute_loss(self, features: Dict[Text, tf.Tensor], training: bool = False) -> tf.Tensor:
        labels = features.pop("rating")
        rating_predictions = self(features)
        # The task computes the loss and the metrics.
        return self.task(
            labels=labels, 
            predictions=rating_predictions
        )


    def fit_model(self, learning_rate: float = 0.1, num_epochs: int = 1) -> None:
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate))
        model.fit(self.cached_train, epochs=num_epochs)


    def evaluate_model(self) -> None:
        self.evaluate(self.cached_test, return_dict=True)


    def predict_model(self, user_id: str, pubs_ids: list[str]) -> None:
        model = self
        test_ratings = {}

        for id in pubs_ids[0]:
            id = id.numpy()
            model_input = self.get_row_as_dict(id)
            model_input['user_id'] = np.array([user_id])
            test_ratings[id] = model(model_input)

        print("Ratings:")
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{title}: {score}")


    def get_row_as_dict(self, id: str) -> Dict:
        id = int(id)
        id = str(id)
        # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
        df_filtered = pubs_df[pubs_df['id'] == id]
        
        # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
        row_as_dict = df_filtered.iloc[0].to_dict()
        
        # Convertir cada valor del diccionario a un numpy.array
        for key in row_as_dict:
            row_as_dict[key] = np.array([row_as_dict[key]])
        
        return row_as_dict
