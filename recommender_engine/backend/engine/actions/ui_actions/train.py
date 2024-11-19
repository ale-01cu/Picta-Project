from engine.data.DataPipeline import DataPipeline
from engine.data import FeaturesTypes
from engine.stages.stages import retrieval_stage, ranking_Stage
from engine.db.cruds.EngineCRUD import EngineCRUD
from engine.db.cruds.ModelCRUD import ModelCRUD
from settings.db import engine
from engine.models.ModelConfig import ModelConfig
import os
import shutil
from engine.actions.data_preprocessing import data_preprocessing
from engine.data.data_preprocessing.transform_date_to_timestamp import transform_date_to_timestamp
from engine.test.movie_lens_datasets import get_datasets
from settings.mongodb import config_collection, engine_collection
from bson import ObjectId
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

def train(engine_id):
    engine = engine_collection.find_one({ "_id": ObjectId(engine_id) })

    if engine['retrieval_model_id']:
        retrieval_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['retrieval_model_id']) })
        
    if engine['ranking_model_id']:
        ranking_config_db = config_collection.find_one(
            { "_id": ObjectId(engine['ranking_model_id']) })

    # General Configs
    engine_name = engine['name']
    is_training_by = engine['is_training_by']
    service_models_path = f"service_models/{engine_name}"

    # Retrieval Configs
    retrieval_config = ModelConfig(
        isTrain=retrieval_config_db['isTrain'],
        model_name=retrieval_config_db['name'],
        features=retrieval_config_db['features'],
        # features=['user_id', 'movie_id', 'bucketized_user_age', 'movie_title', 'timestamp'],
        candidate_data_path=f"../../datasets/{retrieval_config_db['candidate_data_path']}.csv",
        data_path=f"../../datasets/{retrieval_config_db['data_path']}.csv",
        towers_layers_sizes=[],
        shuffle=100_000,
        embedding_dimension=64,
        candidates_batch=128,
        k_candidates=retrieval_config_db['k_candidates'],
        learning_rate=0.12,
        num_epochs=1,
        use_multiprocessing=True,
        workers=4,
        train_batch=16_384,
        val_batch=4096,
        test_batch=4096,
        vocabularies_batch=1000,
        train_length=60,
        test_length=20,
        val_length=20,
        seed=8,
        candidate_feature_merge=retrieval_config_db['candidate_feature_merge'],
        data_feature_merge=retrieval_config_db['data_feature_merge'],
        user_id_data=retrieval_config_db['user_id_data'],
        # user_id_data={ 'user_id': { 'dtype': FeaturesTypes.CategoricalString, 'w': 1 } },
        features_data_q=retrieval_config_db['features_data_q'],
        features_data_c=retrieval_config_db['features_data_c'],
        to_map=True
    )

    print(retrieval_config)
    
    # Likes Configs
    ranking_config = ModelConfig(
        isTrain=ranking_config_db['isTrain'],
        model_name=ranking_config_db['name'],
        features=ranking_config_db['features'],
        candidate_data_path=f"../../datasets/{ranking_config_db['candidate_data_path']}.csv",
        data_path=f"../../datasets/{ranking_config_db['data_path']}.csv",
        towers_layers_sizes=[],
        deep_layers_sizes = [],
        shuffle=154_396,
        embedding_dimension=64,
        learning_rate=0.0001,
        num_epochs=10,
        use_multiprocessing=True,
        target_column=ranking_config_db['target_column'],
        workers=4,
        train_batch=1024,
        val_batch=256,
        test_batch=256,
        vocabularies_batch=512,
        train_length=60,
        test_length=20,
        val_length=20,
        seed=8,
        candidate_feature_merge=ranking_config_db['candidate_feature_merge'],
        data_feature_merge=ranking_config_db['data_feature_merge'],
        user_id_data=ranking_config_db['user_id_data'],
        features_data_q=ranking_config_db['features_data_q'],
        features_data_c=ranking_config_db['features_data_c'],
        to_map=True
    )

    print(ranking_config)

    
    print(f"***** Proceso de Entrenamiendo del Systema {engine_name} Iniciado *****")
 
    if not is_training_by or retrieval_config.isTrain:
        pipe = DataPipeline()
        pubs_df, views_df = pipe.read_csv_data(paths=[
            retrieval_config.candidate_data_path, 
            retrieval_config.data_path
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
        
        # model_crud.turn_off_all(stage=retrieval_stage.name)
        
        config_collection.update_many(
            { '_id': retrieval_config_db['_id'] },
            { '$set': {
                "stage": retrieval_stage.name,
                "modelPath": retrieval_model.model_path,
                "data_train_path": retrieval_model.data_train_path,
                "metadata_path": retrieval_model.model_metadata_path,
                "is_trainned": True,
                "candidates_batch": retrieval_config.candidates_batch,
                "towers_layers_sizes": retrieval_config.towers_layers_sizes,
                "deep_layers_sizes": retrieval_config.deep_layers_sizes,
                "shuffle": retrieval_config.shuffle,
                "embedding_dimension": retrieval_config.embedding_dimension,
                "learning_rate": retrieval_config.learning_rate,
                "num_epochs": retrieval_config.num_epochs,
                "use_multiprocessing": retrieval_config.use_multiprocessing,
                "workers": retrieval_config.workers,
                "train_batch": retrieval_config.train_batch,
                "val_batch": retrieval_config.val_batch,
                "test_batch": retrieval_config.test_batch,
                "vocabularies_batch": retrieval_config.vocabularies_batch,
                "train_length": retrieval_config.train_length,
                "test_length": retrieval_config.test_length,
                "val_length": retrieval_config.val_length,
                "seed": retrieval_config.seed,
            }}
        )

        # model_crud.create(
        #     name=retrieval_model.model_name,
        #     stage=retrieval_stage.name,
        #     model_path=retrieval_model.model_path,
        #     data_train_path=retrieval_model.data_train_path,
        #     metadata_path=retrieval_model.model_metadata_path,
        #     engine_id=new_engine.id
        # )


    if not is_training_by or ranking_config.isTrain:
        pipe = DataPipeline()
        likes_df, pubs_df = pipe.read_csv_data(paths=[
            ranking_config.data_path,
            ranking_config.candidate_data_path
        ])

        pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
        pubs_df['nombre'] = pubs_df['nombre'].astype(str)

        likes_df = likes_df[: ranking_config.shuffle]
        # likes_df = likes_df.drop(['id'], axis=1)
        likes_df[ranking_config.target_column['new']] = likes_df[ranking_config.target_column['current']]\
            .map({True: 1, False: 0})
        # likes_df['fecha'] = likes_df['fecha'].astype("int32")
        

        all_features = ranking_config.features + [ranking_config.target_column['new']]
        likes_df = pipe.merge_data(
            left_data=pubs_df,
            right_data=likes_df,
            left_on=ranking_config.candidate_feature_merge,
            right_on=ranking_config.data_feature_merge,
            output_features=all_features
        )


        #pubs_df['id'] = pubs_df['id'].astype(str)
        #likes_df['id'] = likes_df['id'].astype(str)
        #likes_df['usuario_id'] = likes_df['usuario_id'].astype(str)
        likes_ds = pipe.convert_to_tf_dataset(likes_df)
        
        vocabularies = pipe.build_vocabularies(
            features=all_features,
            ds=likes_ds,
            batch=ranking_config.vocabularies_batch
        )

        total, train_Length, val_length, test_length = pipe.get_lengths(
            ds=likes_ds,
            train_length=ranking_config.train_length,
            test_length=ranking_config.test_length,
            val_length=ranking_config.val_length
        )

        train, val, test = pipe.split_into_train_and_test(
            ds=likes_ds,
            shuffle=ranking_config.shuffle,
            train_length=train_Length,
            val_length=val_length,
            test_length=test_length,
            seed=ranking_config.seed
        )

        cached_train, cached_val, cached_test = pipe.data_caching(
            train=train,
            val=val,
            test=test,
            shuffle=ranking_config.shuffle,
            train_batch=ranking_config.train_batch,
            val_batch=ranking_config.val_batch,
            test_batch=ranking_config.test_batch
        )

        ranking_model = ranking_Stage.model(
            config=ranking_config,
            vocabularies=vocabularies,
            regularization_l2=0.05,
            # model_name=ranking_model_name,
            # towers_layers_sizes=likes_towers_layers_sizes,
            # deep_layers_sizes=likes_deep_layers_sizes,
            # features_data_q=likes_features_data_q,
            # features_data_c=likes_features_data_c,
            # embedding_dimension=likes_embbedding_dimension, 
        )


        ranking_model.fit_model(
            cached_train=cached_train,
            cached_val=cached_val,
            # learning_rate=likes_learning_rate,
            # num_epochs=likes_num_epochs,
            # use_multiprocessing=likes_use_multiprocessing,
            # workers=likes_workers   
        )

        ranking_model.evaluate_model(
            cached_test=cached_test,
            cached_train=cached_train
        )

        ranking_model.save_model(service_models_path, likes_ds)

        config_collection.update_many(
            { '_id': ranking_config_db['_id'] },
            { '$set': {
                "stage": ranking_Stage.name,
                "modelPath": ranking_model.model_path,
                "data_train_path": ranking_model.data_train_path,
                "metadata_path": ranking_model.model_metadata_path,
                "is_trainned": True,
                "towers_layers_sizes": ranking_config.towers_layers_sizes,
                "deep_layers_sizes": ranking_config.deep_layers_sizes,
                "shuffle": ranking_config.shuffle,
                "embedding_dimension": ranking_config.embedding_dimension,
                "learning_rate": ranking_config.learning_rate,
                "num_epochs": ranking_config.num_epochs,
                "use_multiprocessing": ranking_config.use_multiprocessing,
                "workers": ranking_config.workers,
                "train_batch": ranking_config.train_batch,
                "val_batch": ranking_config.val_batch,
                "test_batch": ranking_config.test_batch,
                "vocabularies_batch": ranking_config.vocabularies_batch,
                "train_length": ranking_config.train_length,
                "test_length": ranking_config.test_length,
                "val_length": ranking_config.val_length,
                "seed": ranking_config.seed,
            }}
        )

    #     model_crud.turn_off_all(stage=ranking_Stage.name)
    #     model_crud.create(
    #         name=likes_model.model_name,
    #         stage=ranking_Stage.name,
    #         model_path=likes_model.model_path,
    #         data_train_path=likes_model.data_train_path,
    #         metadata_path=likes_model.model_metadata_path,
    #         engine_id=new_engine.id
    #     )

    # engine_crud.close_session()
    # model_crud.close_session()
