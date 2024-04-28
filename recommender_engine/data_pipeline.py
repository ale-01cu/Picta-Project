import pandas as pd
import tensorflow as tf
import numpy as np

pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
ratings_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/publicaciones_ratings_con_timestamp.csv')

ratings_df = ratings_df.merge(pubs_df, how='inner', left_on='publication_id', right_on='id')[['user_id', 'publication_id', 'timestamp', 'rating','nombre', 'descripcion', 'categoria']]
ratings_df['user_id'] = ratings_df['user_id'].astype(str)
ratings_df['publication_id'] = ratings_df['publication_id'].astype(str)
ratings_df['nombre'] = ratings_df['nombre'].astype(str)
ratings_df['descripcion'] = ratings_df['descripcion'].astype(str)
pubs_df['id'] = pubs_df['id'].astype(str)
pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
pubs_df['nombre'] = pubs_df['nombre'].astype(str)

ratings_ds = tf.data.Dataset.from_tensor_slices(dict(ratings_df))
pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))

pubs_ids_ds = pubs_ds.map(lambda x: x['id'])
pubs_ids_ds_batch = pubs_ids_ds.batch(1_000)
unique_pubs_ids = np.unique(np.concatenate(list(pubs_ids_ds_batch)))

pubs_names_ds = pubs_ds.map(lambda x: x['nombre'])
pubs_names_ds_batch = pubs_names_ds.batch(1_000)
unique_pubs_names = np.unique(np.concatenate(list(pubs_names_ds_batch)))

pubs_descriptions_ds = pubs_ds.map(lambda x: x['descripcion'])
pubs_descriptions_ds_batch = pubs_descriptions_ds.batch(1_000)
unique_pubs_descriptions = np.unique(np.concatenate(list(pubs_descriptions_ds_batch)))

pubs_categories_ds = pubs_ds.map(lambda x: x['categoria'])
pubs_categories_ds_batch = pubs_categories_ds.batch(1_000)
unique_pubs_categories = np.unique(np.concatenate(list(pubs_categories_ds_batch)))

user_ids_ds = ratings_ds.map(lambda x: x['user_id'])
user_ids_ds_batch = user_ids_ds.batch(1_000)
unique_users_ids = np.unique(np.concatenate(list(user_ids_ds_batch)))


pubs_names_batch = pubs_names_ds.batch(128)


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