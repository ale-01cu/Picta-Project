from typing import Optional, Tuple, Dict, Text, List
import collections
import tensorflow as tf
import numpy as np

import pandas as pd

def agrupar_publicaciones(df, listas_por_usuario, publicaciones_por_lista, caracteristicas):
    # Ordenar el DataFrame por 'user_id' y 'timestamp'
    df = df.sort_values(['user_id', 'timestamp'])

    # Crear un diccionario vacío para almacenar los resultados
    resultados = {}

    # Agrupar el DataFrame por 'user_id'
    for user_id, grupo in df.groupby('user_id'):
        # Crear un diccionario vacío para este usuario
        resultados[user_id] = []

        # Dividir el grupo en listas de publicaciones
        publicaciones = [grupo.iloc[i:i+publicaciones_por_lista] for i in range(0, len(grupo), publicaciones_por_lista)]

        # Procesar cada lista de publicaciones
        for lista in publicaciones:
            # Crear un diccionario para esta lista
            lista_dict = {}

            # Extraer las características adicionales
            for caracteristica in caracteristicas:
                lista_dict[caracteristica] = lista[caracteristica].tolist()

            # Extraer los IDs de las publicaciones
            lista_dict['context_pub_id'] = lista['pub_id'].tolist()[:-1]
            lista_dict['label_movie_id'] = lista['pub_id'].tolist()[-1]

            # Añadir la lista al usuario
            resultados[user_id].append(lista_dict)

        # Limitar el número de listas por usuario
        resultados[user_id] = resultados[user_id][:listas_por_usuario]

    return resultados




