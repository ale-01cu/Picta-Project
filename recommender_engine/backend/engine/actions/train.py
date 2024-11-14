from engine.data.DataPipeline import DataPipeline
from engine.data import FeaturesTypes
from engine.stages.stages import retrieval_stage, ranking_Stage
from engine.db.cruds.EngineCRUD import EngineCRUD
from engine.db.cruds.ModelCRUD import ModelCRUD
from engine.db.config import engine
from engine.models.ModelConfig import ModelConfig
import os
import pandas as pd
import shutil
from engine.db.main import build_db

dirname = os.path.dirname(__file__)

# def dinamic_train(config: ModelConfig): 
#     pipe = DataPipeline()
#     data_train_df, = pipe.read_csv_data(paths=config.data_paths)

def delete_path(path):
    path = os.path.join(dirname, f"../{path}")
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error al eliminar engine: {e}")
    else:
        print(f"El engine no existe: {path}")

def train():
    global engine

    # General Configs
    engine_name = "Engine Test #1"
    is_training_by = False
    service_models_path = f"service_models/{engine_name}"

    engine_crud = EngineCRUD(engine=engine)
    model_crud = ModelCRUD(engine=engine)
    new_engine = engine_crud.get_engine_running()

    if not is_training_by: 
        delete_path(service_models_path)
        engine_crud.turn_off_all()
        new_engine = engine_crud.create(name=engine_name)
        model_crud.turn_off_all()


    # Retrieval Configs
    retrieval_config = ModelConfig(
        isTrain=True,
        model_name="Retrieval Model 1M 4F",
        # features=['username', 'fecha', "nombre", 'categoria'],
        features=['username', 'fecha', "nombre", 'categoria'],
        # features=['user_id', 'movie_id', 'bucketized_user_age', 'movie_title', 'timestamp'],
        candidate_data_path="../../datasets/pubs.csv",
        data_path="../../datasets/vistas.csv",
        towers_layers_sizes=[],
        # shuffle=1_000_531,
        shuffle=1000,
        embedding_dimension=128,
        candidates_batch=128,
        k_candidates=100,
        learning_rate=0.1,
        num_epochs=1,
        use_multiprocessing=True,
        workers=4,
        train_batch=64,
        val_batch=32,
        test_batch=32,
        vocabularies_batch=1000,
        train_Length=70,
        test_length=15,
        val_length=15,
        seed=8,
        candidate_feature_merge="id",
        data_feature_merge="publicacion_id",
        user_id_data={ 'username': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 } },
        # user_id_data={ 'user_id': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 } },
        features_data_q={
            # 'edad': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            #'fecha': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 }    
            #'usuario_id': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3 } 
            # 'fecha_nacimiento': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 },    
            'fecha': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 }       
        },
        features_data_c={ 
            # 'nombre': { 'dtype': FeaturesTypes.StringText, 'w': 0.5 },
            #'descripcion': { 'dtype': FeaturesTypes.StringText, 'w': 0.1 }
            'nombre': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 },
            'categoria': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 },
            # 'descripcion': { 'dtype': FeaturesTypes.StringText, 'w': 0.5 }
        }
    )
    
    # Likes Configs
    likes_config = ModelConfig(
        isTrain=False,
        model_name="Likes lite",
        # features=['usuario_id', 'id', 'fecha', 'nombre', 'edad', 'descripcion'],
        features=['username', 'fecha_nacimiento', 'fecha', "nombre", "descripcion", 'categoria'],
        candidate_data_path="../../datasets/pubs.csv",
        data_path="../../datasets/likes.csv",
        towers_layers_sizes=[],
        deep_layers_sizes = [],
        shuffle=1000,
        embedding_dimension=64,
        learning_rate=0.0001,
        num_epochs=1,
        use_multiprocessing=True,
        target_column={
            "current": "valor",
            "new": "like_dislike"
        },
        workers=4,
        train_batch=1024,
        val_batch=256,
        test_batch=256,
        vocabularies_batch=512,
        train_Length=60,
        test_length=20,
        val_length=20,
        seed=8,
        candidate_feature_merge="id",
        data_feature_merge="publicacion_id",
        user_id_data={ 'username': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 } },
        features_data_q={
            # 'edad': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            #'fecha': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 }    
            #'usuario_id': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3 } 
            'fecha_nacimiento': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 },    
            'fecha': { 'dtype': FeaturesTypes.CategoricalContinuous, 'w': 1 }       
        },
        features_data_c={ 
            # 'nombre': { 'dtype': FeaturesTypes.StringText, 'w': 0.5 },
            #'descripcion': { 'dtype': FeaturesTypes.StringText, 'w': 0.1 }
            'nombre': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 },
            'categoria': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 },
            'descripcion': { 'dtype': FeaturesTypes.StringText, 'w': 0.5 }
        }
    )

    
    print(f"***** Proceso de Entrenamiendo del Systema {engine_name} Iniciado *****")
 
    if not is_training_by or retrieval_config.isTrain:
        pipe = DataPipeline()
        pubs_df, views_df = pipe.read_csv_data(paths=[
            retrieval_config.candidate_data_path, 
            retrieval_config.data_path,
        ])    

        # pubs_df = pubs_df[: 5000]

        views_df = views_df[: retrieval_config.shuffle]
        # views_df = views_df.drop(['id'], axis=1)
        # views_df['fecha'] = views_df['fecha'].astype("int32")

        pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
        pubs_df['nombre'] = pubs_df['nombre'].astype(str)


        views_df = pipe.merge_data(
            left_data=pubs_df,
            right_data=views_df,
            left_on=retrieval_config.candidate_feature_merge,
            right_on=retrieval_config.data_feature_merge,
            output_features=retrieval_config.features
        )

        #pubs_df['id'] = pubs_df['id'].apply(lambda x: f'"{x}"'.encode())
        #pubs_df['id'] = pubs_df['id'].astype(str)
        #views_df['id'] = views_df['id'].astype(str)
        #views_df['usuario_id'] = views_df['usuario_id'].apply(str)
        #print(pubs_df['id'])

        pubs_ds = pipe.convert_to_tf_dataset(pubs_df)
        views_ds = pipe.convert_to_tf_dataset(views_df)

        # import tensorflow as tf
        # pubs_ds = pubs_ds.map(lambda x: {**x, "id": tf.strings.as_string(x['id'])})
        # views_ds = views_ds.map(lambda x: {
        #     "id": tf.strings.as_string(int(x['id'])), 
        #     "usuario_id": tf.strings.as_string(int(x['usuario_id']))
        # })

        # for i in pubs_ds.take(1).as_numpy_iterator():
        #     print(i)
        for i in views_ds.take(1).as_numpy_iterator():
            print(i)

        # views_ds, pubs_ds = get_datasets()

        # print(views_ds)
        # print(pubs_ds)

        vocabularies = pipe.build_vocabularies(
            features=retrieval_config.features,
            ds=views_ds,
            batch=retrieval_config.vocabularies_batch
        )

        total, train_Length, val_length, test_length = pipe.get_lengths(
            ds=views_ds,
            train_length=retrieval_config.train_length,
            test_length=retrieval_config.test_length,
            val_length=retrieval_config.val_length
        )

        train, val, test = pipe.split_into_train_and_test(
            ds=views_ds,
            shuffle=retrieval_config.shuffle,
            train_length=train_Length,
            val_length=val_length,
            test_length=test_length,
            seed=retrieval_config.seed
        )

        cached_train, cached_val, cached_test = pipe.data_caching(
            train=train,
            val=val,
            test=test,
            shuffle=retrieval_config.shuffle,
            train_batch=retrieval_config.train_batch,
            val_batch=retrieval_config.val_batch,
            test_batch=retrieval_config.test_batch
        )

        pipe.close()

        retrieval_model = retrieval_stage.model(
            # model_name=retrieval_model_name,
            # towers_layers_sizes=towers_layers_sizes,
            vocabularies=vocabularies,
            regularization_l2=0.05,
            # features_data_q=features_data_q,
            # features_data_c=features_data_c,
            # embedding_dimension=embedding_dimension, 
            # test=test, 
            # shuffle=10_000, 
            # test_batch=512, 
            candidates=pubs_ds,
            # candidates_batch=candidates_batch, 
            # k_candidates=k_candidates
            config=retrieval_config
        )

        retrieval_model.fit_model(
            cached_train=cached_train,
            cached_val=cached_val,
            # learning_rate=learning_rate,
            # num_epochs=num_epochs,
            # use_multiprocessing=use_multiprocessing,
            # workers=workers   
        )
        retrieval_model.evaluate_model(
            cached_test=cached_test,
            cached_train=cached_train
        )
        retrieval_model.save_model(
            service_models_path, views_ds)
        
        model_crud.turn_off_all(stage=retrieval_stage.name)

        model_crud.create(
            name=retrieval_model.model_name,
            stage=retrieval_stage.name,
            model_path=retrieval_model.model_path,
            data_train_path=retrieval_model.data_train_path,
            metadata_path=retrieval_model.model_metadata_path,
            engine_id=new_engine.id
        )


    if not is_training_by or likes_config.isTrain:
        pipe = DataPipeline()
        likes_df, pubs_df = pipe.read_csv_data(paths=[
            likes_config.data_path,
            likes_config.candidate_data_path
        ])

        pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
        pubs_df['nombre'] = pubs_df['nombre'].astype(str)

        likes_df = likes_df[: likes_config.shuffle]
        # likes_df = likes_df.drop(['id'], axis=1)
        likes_df[likes_config.target_column['new']] = likes_df[likes_config.target_column['current']]\
            .map({True: 1, False: 0})
        # likes_df['fecha'] = likes_df['fecha'].astype("int32")
        

        all_features = likes_config.features + [likes_config.target_column['new']]
        likes_df = pipe.merge_data(
            left_data=pubs_df,
            right_data=likes_df,
            left_on=likes_config.candidate_feature_merge,
            right_on=likes_config.data_feature_merge,
            output_features=all_features
        )


        #pubs_df['id'] = pubs_df['id'].astype(str)
        #likes_df['id'] = likes_df['id'].astype(str)
        #likes_df['usuario_id'] = likes_df['usuario_id'].astype(str)
        likes_ds = pipe.convert_to_tf_dataset(likes_df)

        # likes_ds = likes_ds.map(lambda x: {
        #     **x,
        #     "id": tf.strings.as_string(int(x['id'])), 
        #     "usuario_id": tf.strings.as_string(int(x['usuario_id']))
        # })
        
        vocabularies = pipe.build_vocabularies(
            features=all_features,
            ds=likes_ds,
            batch=likes_config.vocabularies_batch
        )

        total, train_Length, val_length, test_length = pipe.get_lengths(
            ds=likes_ds,
            train_length=likes_config.train_length,
            test_length=likes_config.test_length,
            val_length=likes_config.val_length
        )

        train, val, test = pipe.split_into_train_and_test(
            ds=likes_ds,
            shuffle=likes_config.shuffle,
            train_length=train_Length,
            val_length=val_length,
            test_length=test_length,
            seed=likes_config.seed
        )

        cached_train, cached_val, cached_test = pipe.data_caching(
            train=train,
            val=val,
            test=test,
            shuffle=likes_config.shuffle,
            train_batch=likes_config.train_batch,
            val_batch=likes_config.val_batch,
            test_batch=likes_config.test_batch
        )

        likes_model = ranking_Stage.model(
            config=likes_config,
            vocabularies=vocabularies,
            regularization_l2=0.05,
            # model_name=likes_model_name,
            # towers_layers_sizes=likes_towers_layers_sizes,
            # deep_layers_sizes=likes_deep_layers_sizes,
            # features_data_q=likes_features_data_q,
            # features_data_c=likes_features_data_c,
            # embedding_dimension=likes_embbedding_dimension, 
        )


        likes_model.fit_model(
            cached_train=cached_train,
            cached_val=cached_val,
            # learning_rate=likes_learning_rate,
            # num_epochs=likes_num_epochs,
            # use_multiprocessing=likes_use_multiprocessing,
            # workers=likes_workers   
        )

        likes_model.evaluate_model(
            cached_test=cached_test,
            cached_train=cached_train
        )

        likes_model.save_model(service_models_path, likes_ds)

        model_crud.turn_off_all(stage=ranking_Stage.name)
        model_crud.create(
            name=likes_model.model_name,
            stage=ranking_Stage.name,
            model_path=likes_model.model_path,
            data_train_path=likes_model.data_train_path,
            metadata_path=likes_model.model_metadata_path,
            engine_id=new_engine.id
        )

    engine_crud.close_session()
    model_crud.close_session()


if __name__ == "__main__":
    build_db()
    train()