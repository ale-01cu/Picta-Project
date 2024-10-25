import tensorflow as tf
import numpy as np
import os
from engine.db.cruds.ModelCRUD import ModelCRUD
from engine.stages.stages import retrieval_stage, ranking_Stage
from settings.db import engine
from engine.utils import read_json
from engine.data.FeaturesTypes import map_feature
from engine.data.DataPipeline import DataPipeline
from settings.mongodb import engine_collection, config_collection
from bson import ObjectId
import time
dirname = os.path.dirname(__file__)

engine = engine_collection.find_one({ "is_active": True })

def get_recommendations(user_id, k, params):
    retrieval_config_db = config_collection.find_one(
        { "_id": ObjectId(engine['retrieval_model_id']) })

    metadata = read_json(retrieval_config_db["metadata_path"])

    retieval_model = tf.saved_model.load(
        os.path.join(dirname, f"{retrieval_config_db['modelPath']}/index"))
    
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

    if 'fecha' in model_input:
        model_input['fecha'] = tf.constant([params['fecha']], dtype=tf.int64)
    if 'timestamp' in model_input:
        model_input['timestamp'] = tf.constant([params['timestamp']], dtype=tf.int64)

    for key, value in user_id_data.items():
        model_input[key] = value

    #input_tensor = tf.constant([user_id], dtype=tf.int32)
    scores, ids = retieval_model(
        model_input, 
        training=False
    )


    for id, score in zip(ids.numpy()[0], scores.numpy()[0]):
        yield id, score


def ranking_recommendations(user_id, pubs_ids: list, params: dict): 
    ranking_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['ranking_model_id']) })

    metadata = read_json(ranking_config_db['metadata_path'])
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

    if 'fecha' in context_input:
        context_input['fecha'] = tf.constant([params['fecha']], dtype=tf.int32)

    for key, value in user_id_data.items():
        context_input[key] = value
    
    model = tf.saved_model.load(
        os.path.join(dirname, f"{ranking_config_db['modelPath']}/service"))
    
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
    #data_df = transform_date_to_timestamp(data_df, 'fecha')
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
    retrieval_config_db = config_collection.find_one(
        { "_id": ObjectId(engine['retrieval_model_id']) })

    metadata = read_json(retrieval_config_db["metadata_path"])

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

    # response = []

    # for i in results:
    #     response.append({ "id": i[0].item() })
    # # print("Top ", k, " recomendaciones para el usuario ", user_id)
    # # for id, score in recommendations:
    # #     response.append({ "id": id.item() })
    #     # print(id)
    #     # print(get_row_as_dict(id), "Score: ", score)

    # return response[: k]


data_test = [
    {
        "usuario_id": np.array([320]),
        "edad": np.array([32])
    },
    {
        "usuario_id": np.array([161]),
        "edad": np.array([37])
    },
    {
        "usuario_id": np.array([3040]),
        "edad": np.array([51])
    },
    {
        "usuario_id": np.array([8097]),
        "edad": np.array([45])
    },
    {
        "usuario_id": np.array([9364]),
        "edad": np.array([22])
    },
]
data_test = [
    {
        "user_id": np.array([b'138']),
        "bucketized_user_age": np.array([45])
    },
    {
        "user_id": np.array([b'92']),
        "bucketized_user_age": np.array([25])
    },
    {
        "user_id": np.array([b'301']),
        "bucketized_user_age": np.array([18])
    },
    {
        "user_id": np.array([b'60']),
        "bucketized_user_age": np.array([50])
    },
    {
        "user_id": np.array([b'197']),
        "bucketized_user_age": np.array([50])
    },
]


if __name__ == "__main__":
    data = data_test[0]
    #USER_ID = data['usuario_id'].item()
    USER_ID = data['user_id'].item()
    K = 10
    params = {
        "bucketized_user_age": data['bucketized_user_age'].item(),
        'timestamp': int(time.time() * 1000)#+ (7 * 24 * 60 * 60 * 1000)

    }

    print(params)
    res = use_models(user_id=USER_ID, k=K, params=params)
    print(res)

# id  b'323'
# id  b'117'
# id  b'332'
# id  b'301'
# id  b'245'
# id  b'682'
# id  b'313'
# id  b'293'
# id  b'751'
# id  b'547'


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