from typing import Optional, Tuple, Dict, Text, List
import collections
import tensorflow as tf
import numpy as np

import pandas as pd

def group(ds: tf.data.Dataset, q_features: list[str], c_features: list[str], features: list[str]):

    sampled_fetures = { feature: [] for feature in q_features + c_features }

    # Crear un diccionario vacío para almacenar los resultados
    unique_ids = set(np.unique(list(ds.map(lambda x: x['user_id']))))
    
    for user_id in unique_ids:
        print(user_id)
        sequence = ds.filter(lambda x: x['user_id'] == user_id)


        for q, c, f in zip(q_features, c_features, features):
            sequence_data = list(sequence.map(lambda x: x[f]))
            sampled_fetures[q].append(tf.stack(sequence_data[:-1], 0))
            sampled_fetures[c].append(tf.stack(sequence_data[-1:], 0))


    print(sampled_fetures)
#     # ds = tf.data.Dataset.from_tensor_slices(sampled_fetures)
#     # for i in ds.take(1):
#     #     print(i)
        

# def group2(df: tf.data.Dataset, q_features: list[str], c_features: list[str], features: list[str]):

#     sampled_fetures = { feature: [] for feature in q_features + c_features }

#     # Crear un diccionario vacío para almacenar los resultados
#     unique_ids = set(np.unique(list(ds.map(lambda x: x['user_id']))))
    
#     for user_id in unique_ids:
#         print(user_id)
#         sequence = ds.filter(lambda x: x['user_id'] == user_id)


#         for q, c, f in zip(q_features, c_features, features):
#             sequence_data = list(sequence.map(lambda x: x[f]))
#             sampled_fetures[q].append(tf.stack(sequence_data[:-1], 0))
#             sampled_fetures[c].append(tf.stack(sequence_data[-1:], 0))


#     print(sampled_fetures)





