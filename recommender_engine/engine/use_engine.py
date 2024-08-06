import tensorflow as tf
import numpy as np
import pandas as pd
from utils import read_json
import os
dirname = os.path.dirname(__file__)

# data = read_json("./models/info.json")

retieval_model = tf.saved_model.load(
    os.path.join(dirname, "models/Retrieval_Lite_534K_2024-08-05_224503797704/index"))
# positive_model = tf.saved_model.load(BASE_PATH + "/" + data["positive_model_name"])


pubs_path = '../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
pubs_df = pd.read_csv(pubs_path)
pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
pubs_df['nombre'] = pubs_df['nombre'].astype(str)


def get_recommendations(user_id, k):
    input_tensor = tf.constant([user_id], dtype=tf.int32)
    scores, ids = retieval_model(
        {'usuario_id': input_tensor}, training=False)
    
    for id, score in zip(ids.numpy()[0][: k], scores.numpy()[0][: k]):
        yield id, score


# def ranking_recommendations(user_id, pubs_ids: list): 
#     model = positive_model
#     test_ratings = {}

#     for id in pubs_ids:
#         model_input = get_row_as_dict(id)
#         model_input['usuario_id'] = np.array([user_id])

#         model_input = {
#             'id': tf.constant([id], dtype=tf.int64),
#             'usuario_id': tf.constant([user_id], dtype=tf.int64)
#         }
#         predictions = model(model_input, training=False)
#         prediction = sum(predictions.numpy()[0]) / 300
#         test_ratings[id] = prediction

#     result = sorted(test_ratings.items(), key=lambda x: x[1], reverse=True)
#     return result


def get_row_as_dict(id: str, columns=["id", "nombre"]):
    # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
    df_filtered = pubs_df[pubs_df['id'] == id][columns]
    
    # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
    row_as_dict = df_filtered.iloc[0].to_dict()
    
    # Convertir cada valor del diccionario a un numpy.array
    for key in row_as_dict:
        row_as_dict[key] = np.array([row_as_dict[key]])
    
    return row_as_dict


def use_models(user_id, k):
    recommendations = get_recommendations(user_id=user_id, k=k)
    # ids = [id for id, _ in recommendations]
    # results = ranking_recommendations(user_id, ids)
    
    print("Top ", K, " recomendaciones para el usuario ", USER_ID)
    for id, score in recommendations:
        print(id)
        print(get_row_as_dict(id), "Score: ", score)


if __name__ == "__main__":
    USER_ID = 15
    K = 10
    use_models(user_id=USER_ID, k=K)

