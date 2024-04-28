import tensorflow as tf
import tensorflow_recommenders as tfrs
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from typing import Dict, Text
from .data_pipeline import train, test, total, pubs_df
import numpy as np

class RankingModel(tfrs.models.Model):

    def __init__(self, layer_sizes):
        super().__init__()

        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)

        # Compute predictions.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, inputs):
        user_embedding = self.query_model(inputs)
        pub_embedding = self.candidate_model(inputs)

        return self.rating_model(tf.concat([user_embedding, pub_embedding], axis=1))


    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("rating")
        rating_predictions = self(features)
        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)


    def fit_model(self):
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        cached_train = train.shuffle(total).batch(2048).cache()
        cached_test = test.batch(1024).cache()

        num_epoch = 3
        model.fit(cached_train, epochs=num_epoch)

        model.evaluate(cached_test, return_dict=True)

    def predict(self, user_id, titles):
        model = self
        test_ratings = {}
        for movie_title in titles:
            model_input = self.get_row_as_dict(movie_title)
            model_input['user_id'] = np.array([user_id])
            test_ratings[movie_title] = model(model_input)

        print("Ratings:")
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{title}: {score}")


    def get_row_as_dict(self, id):
        print(id)
        # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
        df_filtered = pubs_df[pubs_df['id'] == id]
        
        # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
        row_as_dict = df_filtered.iloc[0].to_dict()
        
        # Convertir cada valor del diccionario a un numpy.array
        for key in row_as_dict:
            row_as_dict[key] = np.array([row_as_dict[key]])
        
        return row_as_dict
