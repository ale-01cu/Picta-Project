from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print('No GPU found')


import pandas as pd
pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
ratings_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/publicaciones_ratings_con_timestamp.csv')


ratings_df = ratings_df.merge(pubs_df, how='inner', left_on='publication_id', right_on='id')[['user_id', 'publication_id', 'nombre']]
ratings_df['user_id'] = ratings_df['user_id'].astype(str)
ratings_df['nombre'] = ratings_df['nombre'].astype(str)
pubs_df['nombre'] = pubs_df['nombre'].astype(str)

ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df))


import math

total = len(ratings_ds)
train_Length = math.ceil(total * (80 / 100))
test_length = int(total * (20 / 100))

print('Total ', total)
print('Tamaño del set de entrenamiento ', train_Length)
print('Tamaño del set de prueba ', test_length)


tf.random.set_seed(42)
shuffled = ratings_ds.shuffle(total, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(train_Length)
test = shuffled.skip(train_Length).take(test_length)



pubs_names_ds = tf.data.Dataset.from_tensor_slices(pubs_df['nombre'].values[:20_000])
pubs_names_ds_batch = pubs_names_ds.batch(1_000)
user_ids_ds = tf.data.Dataset.from_tensor_slices(ratings_df['user_id'].astype(str).values).batch(1_000)

unique_publications_names = np.unique(np.concatenate(list(pubs_names_ds_batch)))
unique_users_ids = np.unique(np.concatenate(list(user_ids_ds)))


class Model(tfrs.Model):
    def __init__(self):
        super().__init__()
        self.embedding_dimension = 32
        self.max_tokens = 10_000

        self.movie_model: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_publications_names, mask_token=None),
            tf.keras.layers.Embedding(len(unique_publications_names) + 1, self.embedding_dimension)
        ])

        self.user_model: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_users_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_users_ids) + 1, self.embedding_dimension)
        ])

        self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=pubs_names_ds.batch(128).map(self.movie_model))
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["nombre"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)
    


model = Model()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))