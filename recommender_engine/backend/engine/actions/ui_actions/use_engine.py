import tensorflow as tf
import numpy as np
import os
from settings.db import engine
from engine.utils import read_json
from engine.data.FeaturesTypes import map_feature
from engine.data.DataPipeline import DataPipeline
from settings.mongodb import engine_collection, config_collection
from bson import ObjectId
from engine.exceptions.use_engine import (
    DataInvalid,
    FeatureNotFound,
    MetadataNotFound,
    ContextDataToTensor,
    ModelInference,
    UserIdToTensor
)
from engine.exceptions.train import (
    ModelIdNotProvide,
    ModelNotFound
)
dirname = os.path.dirname(__file__)
engine = engine_collection.find_one({ "is_active": True })


def get_recommendations(user_id, params):
    retrieval_config_db = config_collection.find_one(
        { "_id": ObjectId(engine['retrieval_model_id']) })

    metadata = read_json(retrieval_config_db["metadata_path"])

    retieval_model = tf.saved_model.load(
        os.path.join(dirname, f"{retrieval_config_db['modelPath']}/index"))
    
    user_identier_name = next(iter(metadata['user_id_data']))
    

    try:
        user_id_data = {
            key: tf.constant(
                [user_id], 
                dtype=map_feature(to_class=True, feature_type=value['dtype'])().datatype,
                name=key
            ) 
            for key, value in metadata['user_id_data'].items() 
        }
    except:
        raise UserIdToTensor.UserIdToTensorException()

    try:
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
    except:
        raise ContextDataToTensor.ContextDataToTensorException()

    # if 'fecha' in model_input:
    #     model_input['fecha'] = tf.constant([params['fecha']], dtype=tf.int64)
    # if 'timestamp' in model_input:
    #     model_input['timestamp'] = tf.constant([params['timestamp']], dtype=tf.int64)

    # for key, value in user_id_data.items():
    #     model_input[key] = value

    model_input = {
        **user_id_data,
        **model_input
    }

    #input_tensor = tf.constant([user_id], dtype=tf.int32)
    try:
        _, ids = retieval_model(
            model_input, 
            training=False
        )
    except:
        raise ModelInference.ModelInferenceException("Retrieval")

    for id in ids.numpy()[0]:
        yield id


def ranking_recommendations(user_id, pubs_ids: GeneratorExit, params: dict): 
    ranking_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['ranking_model_id']) })

    metadata = read_json(ranking_config_db['metadata_path'])
    user_identier_name = next(iter(metadata['user_id_data']))

    feature_data_merged = {
        **metadata['features_data_q'],
        **metadata['features_data_c']
    }

    user_id_data = {
        key: tf.constant(
            [user_id], 
            dtype=(tf.int64 if map_feature(
                to_class=True, 
                feature_type=value['dtype']
            )().datatype == tf.int32 
                   else map_feature(
                       to_class=True, 
                       feature_type=value['dtype']
                    )().datatype),
            name=key
        )
        for key, value in metadata['user_id_data'].items()
    }
    
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

    context_input = {
        **user_id_data,
        **context_input
    }

    
    model = tf.saved_model.load(
        os.path.join(dirname, f"{ranking_config_db['modelPath']}/service"))
    
    test_ratings = {}

    full_data_trained = get_full_data(metadata)
    # count = 1
    for id in pubs_ids:
        # print("Number: ", count)
        # count += 1
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
        output_features=hiperparams['features'] + [hiperparams['candidate_feature_merge']] + [hiperparams['data_feature_merge']]
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


def validate_inputs(user_id, params):
# Recuperando datos
    if engine['retrieval_model_id']:
        retrieval_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['retrieval_model_id']) })
        if not retrieval_config_db:
            raise ModelNotFound.ModelNotFoundException("Retrieval")
    else: raise ModelIdNotProvide.ModelIdNotProvideException("Retrieval")
    if engine['ranking_model_id']:
        ranking_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['ranking_model_id']) })
        if not ranking_config_db:
            raise ModelNotFound.ModelNotFoundException(engine['ranking_model_id'])
    else:
        raise ModelIdNotProvide.ModelIdNotProvideException("Ranking")
    

    try: retrieval_hyperparams = read_json(retrieval_config_db['metadata_path'])
    except: raise MetadataNotFound.MetadataNotFoundException("Retrieval")
    try: ranking_hyperparams = read_json(ranking_config_db['metadata_path'])
    except: raise MetadataNotFound.MetadataNotFoundException("Ranking")

    user_key = retrieval_hyperparams['user_id_data']
    retrieval_features_data_q = retrieval_hyperparams['features_data_q']
    ranking_feature_data_q = ranking_hyperparams['features_data_q']

    if not user_id:
        raise FeatureNotFound(user_key.keys()[0])

    features_data = {
        **retrieval_features_data_q, 
        **ranking_feature_data_q
    }

    print(params)
    # Actualizar los tipos de datos en user_key y features_data
    for k, v in user_key.items():
        v['dtype'] = map_feature(
            to_class=True, 
            feature_type=v['dtype']
        )
    for key, value in features_data.items():
        if key not in params.keys() and key not in user_key.keys():
            raise FeatureNotFound.FeatureNotFoundException(key)
        value['dtype'] = map_feature(
            to_class=True, 
            feature_type=value['dtype']
        )


    # Validacion
    if len(features_data) > 1:
        # validacion de los parametros
        for key, value in params.items():
            if key not in features_data.keys() and key not in user_key.keys():
                raise FeatureNotFound.FeatureNotFoundException(key)

            dtype = features_data[key]['dtype']()
            try:
                params[key] = dtype.cast(value)
            except:
                raise DataInvalid.DataInvalidException()


    # Validacion de la clave del usuario
    for key, value in user_key.items():
        dtype = value['dtype']()
        user_id = dtype.cast(user_id)

    return user_id, params


def use_models(user_id, k, params):
    # print(params)
    user_id, params = validate_inputs(user_id, params)

    recommendations = get_recommendations(
        user_id=user_id, params=params)

    # print("Recomendados por recuperacion...............")
    # for id in ids[: 10]:
    #     print("id ", id)
    results = ranking_recommendations(user_id, recommendations, params)
    return [id for id, _  in results]
    # for id, score in results:
    #     print("id ", id, "Score ", score)