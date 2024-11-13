import tensorflow as tf
import numpy as np
import os
from engine.db.cruds.ModelCRUD import ModelCRUD
from engine.stages.stages import retrieval_stage, ranking_Stage
from engine.db.config import engine
from engine.db.main import build_db
from engine.utils import read_json
from engine.data.FeaturesTypes import map_feature
from engine.data.DataPipeline import DataPipeline
import time
dirname = os.path.dirname(__file__)


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

    # if 'fecha' in model_input:
    #     model_input['fecha'] = tf.constant([params['fecha']], dtype=tf.int64)
    # if 'timestamp' in model_input:
    #     model_input['timestamp'] = tf.constant([params['timestamp']], dtype=tf.int64)


    for key, value in user_id_data.items():
        model_input[key] = value

    for key in model_input:
        model_input[key] = tf.reshape(model_input[key], [-1])

    #input_tensor = tf.constant([user_id], dtype=tf.int32)
    scores, ids = retieval_model(
        model_input, 
        training=False
    )


    for id, score in zip(ids.numpy()[0], scores.numpy()[0]):
        yield id, score


def ranking_recommendations(user_id, pubs_ids: list, params: dict): 
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
    
    context_input = {
        key: tf.constant(
            [params[key]], 
            dtype=tf.int64 
                    if map_feature(
                        to_class=True, 
                        feature_type=feature_data_merged[key]['dtype'])().datatype == tf.int32 
                    else map_feature(
                        to_class=True, 
                        feature_type=feature_data_merged[key]['dtype'])().datatype,
            name=key
        )
        for key in metadata['features_data_q'].keys() if key != user_identier_name
    }

    # if 'fecha' in context_input:
    #     context_input['fecha'] = tf.constant([params['fecha']], dtype=tf.int32)

    for key, value in user_id_data.items():
        context_input[key] = value
    
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

        for key, value in context_input.items():
            model_input[key] = value

        for key in model_input:
            model_input[key] = tf.reshape(model_input[key], [-1])


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


from engine.data.data_preprocessing.transform_date_to_timestamp import transform_date_to_timestamp
def get_full_data(hiperparams):
    candidate_path, data_path = (
        "../../../../datasets/picta_publicaciones_crudas.csv", 
        hiperparams['data_path']
    )

    pipe = DataPipeline()
    candidate_df, data_df = pipe.read_csv_data(paths=[
        candidate_path,
        data_path
    ])
    # data_df = transform_date_to_timestamp(data_df, 'fecha')
    # pubs_path = '../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
    # pubs_df = pd.read_csv(pubs_path)
    candidate_df['descripcion'] = candidate_df['descripcion'].astype(str)
    candidate_df['nombre'] = candidate_df['nombre'].astype(str)

    common_columns = set(candidate_df.columns) & set(data_df.columns)
    data_df = data_df.drop(list(common_columns), axis=1)

    merge_feature = hiperparams['candidate_feature_merge']
    hiperparams['features'].append(merge_feature)

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

    # try:
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
    # except Exception as e:
    #     print(e)

    return None


def use_models(user_id, k, params):
    start_time = time.time()  # Inicia el contador de tiempo

    print(params)
    recommendations = get_recommendations(
        user_id=user_id, k=k, params=params)
    
    ids = [id for id, _ in recommendations]
    print("Recomendados por recuperacion...............")
    for id in ids[: 10]:
        print("id ", id)
    results = ranking_recommendations(user_id, ids, params)

    print("Recomendaciones por clasificacion............")
    for id, score in results:
        print("id ", id, "Score ", score)

    # Calcula el tiempo total de ejecución
    elapsed_time = time.time() - start_time
    print(f"Tiempo de ejecución de use_models: {elapsed_time:.2f} segundos")

    # response = []

    # for i in results:
    #     response.append({ "id": i[0].item() })
    # # print("Top ", k, " recomendaciones para el usuario ", user_id)
    # # for id, score in recommendations:
    # #     response.append({ "id": id.item() })
    #     # print(id)
    #     # print(get_row_as_dict(id), "Score: ", score)

    # return response[: k]


import pickle
if __name__ == "__main__":
    build_db()
    retrieval_model_db = model_crud.get_model_running_by_stage(
        stage=retrieval_stage.name
    )
    ranking_model_db = model_crud.get_model_running_by_stage(
        stage=ranking_Stage.name
    )
    retrieval_data_path = retrieval_model_db.data_train_path
    ranking_data_path = ranking_model_db.data_train_path

    retrieval_hyperparams = read_json(retrieval_model_db.metadata_path)
    ranking_hyperparams = read_json(ranking_model_db.metadata_path)

    pipe = DataPipeline()
    df, = pipe.read_csv_data(paths=[retrieval_hyperparams['data_path']])

    features_data = list(set(
        key for sublist in [
            list(retrieval_hyperparams['features_data_q'].keys()), 
            list(ranking_hyperparams['features_data_q'].keys())
        ] for key in sublist
    ))

    data_test = [{
        key: np.array([df[key].sample(n=1).iloc[0]])
        for key in features_data
    } for _ in range(5)]
    
    params = data_test[0]

    K = 10
    print("paramssssssssssss")
    print(params)
    res = use_models(
        user_id=params[list(retrieval_hyperparams['user_id_data'].keys())[0]], 
        k=K, 
        params=params
    )
    print(res)