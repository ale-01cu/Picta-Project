import tensorflow as tf
import numpy as np
import pandas as pd
from apps.recommender.utils import read_json
from apps.pubs.models import PubsModel
import os

dirname = os.path.dirname(__file__)


data = read_json(os.path.join(dirname, "models/info.json"))

BASE_PATH = os.path.join(dirname, "models")
retrieval_path = os.path.join(BASE_PATH, data["retrieval_model_name"])

retieval_model = tf.saved_model.load(os.path.join(retrieval_path, "index"))
positive_model = tf.saved_model.load(os.path.join(BASE_PATH, data["positive_model_name"]))




pubs_path = PubsModel().db_filename
pubs_df = pd.read_csv(pubs_path)
pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
pubs_df['nombre'] = pubs_df['nombre'].astype(str)


def get_recommendations(user_id, k):
    input_tensor = tf.constant([user_id], dtype=tf.int32)
    scores, ids = retieval_model(
        {'usuario_id': input_tensor}, training=False)
    
    for id, score in zip(ids.numpy()[0][: k], scores.numpy()[0][: k]):
        yield id, score


def ranking_recommendations(user_id, pubs_ids: list): 
    model = positive_model
    test_ratings = {}

    for id in pubs_ids:
        model_input = get_row_as_dict(id)
        model_input['usuario_id'] = np.array([user_id])

        model_input = {
            'id': tf.constant([id], dtype=tf.int64),
            'usuario_id': tf.constant([user_id], dtype=tf.int64)
        }
        predictions = model(model_input, training=False)
        prediction = sum(predictions.numpy()[0]) / 300
        test_ratings[id] = prediction

    result = sorted(test_ratings.items(), key=lambda x: x[1], reverse=True)
    return result


def get_row_as_dict(id: str, columns=["id", "nombre"]):
    # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
    df_filtered = pubs_df[pubs_df['id'] == id][columns]
    
    # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
    row_as_dict = df_filtered.iloc[0].to_dict()
    
    # Convertir cada valor del diccionario a un numpy.array
    for key in row_as_dict:
        row_as_dict[key] = np.array([row_as_dict[key]])
    
    return row_as_dict


def engine(user_id, k):
    recommendations = get_recommendations(user_id=user_id, k=k)
    ids = [id[0] for id in recommendations]
    results = ranking_recommendations(user_id, ids)
    
    # print("Top ", k, " recomendaciones para el usuario ", user_id)
    # for id, score in results:
    #     print(get_row_as_dict(id), "Score: ", score)

    return [get_row_as_dict(id, columns=["id", "nombre", "descripcion", "categoria"]) for id, score in results]


if __name__ == "__main__":
    USER_ID = 111111111
    K = 100
    engine(user_id=USER_ID, k=K)
