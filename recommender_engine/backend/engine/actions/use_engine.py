import tensorflow as tf
import numpy as np
import pandas as pd
import os
from engine.db.cruds.ModelCRUD import ModelCRUD
from engine.stages.stages import retrieval_stage, ranking_Stage
from settings.db import engine
from engine.utils import read_json
from engine.models.ModelConfig import ModelConfig
from engine.FeaturesTypes import map_feature
from engine.DataPipeline import DataPipeline
dirname = os.path.dirname(__file__)
from engine.db.main import build_db

build_db()

model_crud = ModelCRUD(engine=engine)

def get_recommendations(user_id, k, params):
    retrieval_model_db = model_crud.get_model_running_by_stage(
        stage=retrieval_stage.name)
    metadata = read_json(retrieval_model_db.metadata_path)

    retieval_model = tf.saved_model.load(
        os.path.join(dirname, f"{retrieval_model_db.model_path}/index"))
    
    user_identier_name = next(iter(metadata['user_id_data']))
    

    user_id_data = {
        key: tf.constant(
            [user_id], 
            dtype=map_feature(to_class=True, feature_type=value['dtype'])().datatype,
            name=key
        ) 
        for key, value in metadata['user_id_data'].items() 
    }

    model_input = {
        key: tf.constant(
            [params[key]], 
            dtype=map_feature(
                to_class=True, 
                feature_type=metadata['features_data_q'][key]['dtype']
            )().datatype,
            name=key
        )
        for key in metadata['features_data_q'].keys() if key != user_identier_name
    }

    for key, value in user_id_data.items():
        model_input[key] = value

    #input_tensor = tf.constant([user_id], dtype=tf.int32)
    scores, ids = retieval_model(
        model_input, 
        training=False
    )
    
    for id, score in zip(ids.numpy()[0][: k], scores.numpy()[0][: k]):
        yield id, score


def ranking_recommendations(user_id, pubs_ids: list): 
    likes_model_db = model_crud.get_model_running_by_stage(
        stage=ranking_Stage.name)
    
    metadata = read_json(likes_model_db.metadata_path)
    user_identier_name = next(iter(metadata['user_id_data']))

    feature_data_merged = {
        **metadata['features_data_q'],
        **metadata['features_data_c']
    }

    user_id_data = {}
    for key, value in metadata['user_id_data'].items():
        datatype = None 
        if map_feature(to_class=True, feature_type=value['dtype'])().datatype == tf.int32:
            datatype = tf.int64
        else:
            datatype = map_feature(to_class=True, feature_type=value['dtype'])().datatype
        user_id_data[key] = tf.constant(
            [user_id], 
            dtype=datatype,
            name=key
        )
    
    
    model = tf.saved_model.load(
        os.path.join(dirname, f"{likes_model_db.model_path}/service"))
    
    test_ratings = {}

    full_data_trained = get_full_data(metadata)
    for id in pubs_ids:
        row_data = get_row_as_dict(id, full_data_trained)
        if not row_data: continue
        # model_input['usuario_id'] = np.array([user_id])

        features_data_q_keys = [key for key in metadata['features_data_q'] if key != user_identier_name]
        features_data_q_data = {key: row_data[key] for key in features_data_q_keys if key in row_data}

        features_data_c_keys = [key for key in metadata['features_data_c']]
        features_data_c_data = {key: row_data[key] for key in features_data_c_keys if key in row_data}

        model_input = {
            **features_data_q_data,
            **features_data_c_data
        }

        model_input = {
            key: tf.constant(
                [value.item()], 
                dtype=tf.int64 
                    if map_feature(
                        to_class=True, 
                        feature_type=feature_data_merged[key]['dtype'])().datatype == tf.int32 
                    else map_feature(
                        to_class=True, 
                        feature_type=feature_data_merged[key]['dtype'])().datatype,
                name=key
            )
            for key, value in model_input.items()
        }

        for key, value in user_id_data.items():
            model_input[key] = value


        #model_input = {
        #     'id': tf.constant([id], dtype=tf.int64),
        #     'usuario_id': tf.constant([user_id], dtype=tf.int64)
        # }
        predictions = model(model_input, training=False)
        prediction = sum(predictions.numpy()[0]) / 300
        test_ratings[id] = prediction
 
    result = sorted(
        test_ratings.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    return result


def get_full_data(hiperparams):
    candidate_path, data_path = hiperparams['candidate_data_path'], hiperparams['data_path']
    pipe = DataPipeline()
    candidate_df, data_df = pipe.read_csv_data(paths=[
        candidate_path,
        data_path
    ])
    # pubs_path = '../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
    # pubs_df = pd.read_csv(pubs_path)
    candidate_df['descripcion'] = candidate_df['descripcion'].astype(str)
    candidate_df['nombre'] = candidate_df['nombre'].astype(str)

    common_columns = set(candidate_df.columns) & set(data_df.columns)
    data_df = data_df.drop(list(common_columns), axis=1)


    full_data = pipe.merge_data(
        left_data=candidate_df,
        right_data=data_df,
        left_on=hiperparams['candidate_feature_merge'],
        right_on=hiperparams['data_feature_merge'],
        output_features=hiperparams['features']
    )

    pipe.close()
    return full_data



def get_row_as_dict(id: str, data: dict):
    retrieval_model_db = model_crud.get_model_running_by_stage(
        stage=retrieval_stage.name)
    metadata = read_json(retrieval_model_db.metadata_path)

    try:
        # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
        df_filtered = data[data[metadata['candidate_feature_merge']] == id]
        
        # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
        row_as_dict = df_filtered.iloc[0].to_dict()
        
        # Convertir cada valor del diccionario a un numpy.array
        for key in row_as_dict:
            row_as_dict[key] = np.array([row_as_dict[key]])
        
        return row_as_dict

    except Exception as e:
        print(e)

    return None


def use_models(user_id, k, params):
    print(params)
    recommendations = get_recommendations(user_id=user_id, k=k, params=params)
    ids = [id for id, _ in recommendations]
    results = ranking_recommendations(user_id, ids)
    response = []

    for i in results:
        response.append({ "id": i[0].item() })
    # print("Top ", k, " recomendaciones para el usuario ", user_id)
    # for id, score in recommendations:
    #     response.append({ "id": id.item() })
        # print(id)
        # print(get_row_as_dict(id), "Score: ", score)

    return response

if __name__ == "__main__":
    USER_ID = 2005
    K = 10
    use_models(user_id=USER_ID, k=K, params={
        "id": 74,
        'fecha':'2018-04-09%2021:29:59.769471+02:00'
    })



# import tensorflow as tf

# features_data_q = {
#     'usuario_id': {'dtype': 'CategoricalInteger', 'w': 1},
#     'timestamp': {'dtype': 'CategoricalContinuous', 'w': 0.3}
# }

# features_data_c = {
#     'id': {'dtype': 'CategoricalInteger', 'w': 1},
#     'nombre': {'dtype': 'StringText', 'w': 0.2},
#     'descripcion': {'dtype': 'StringText', 'w': 0.1}
# }

# data = {
#     'id': 1,
#     'usuario_id': 2,
#     'timestamp': 3,
#     'nombre': 'Juan',
#     'descripcion': 'Este es un ejemplo'
# }

# # Extraer los datos relevantes de features_data_q
# features_data_q_keys = [key for key in features_data_q if key != 'usuario_id']
# features_data_q_data = {key: data[key] for key in features_data_q_keys if key in data}

# # Extraer los datos relevantes de features_data_c
# features_data_c_keys = [key for key in features_data_c]
# features_data_c_data = {key: data[key] for key in features_data_c_keys if key in data}

# # Construir el objeto model_input
# model_input = {
#     **features_data_q_data,
#     **features_data_c_data
# }

# # Convertir los valores a tf.constant
# model_input = {key: tf.constant([value], dtype=tf.int64) for key, value in model_input.items()}

# print(model_input)