from engine.data.DataPipeline import DataPipeline
from engine.stages.stages import retrieval_stage, ranking_Stage
from engine.models.ModelConfig import ModelConfig
from settings.mongodb import engine_collection, config_collection
from bson import ObjectId
from engine.exceptions.train import (
    EngineNotFound,
    ModelIdNotProvide,
    ModelNotFound,
    UpdateDB
)

def fine_tunning(engine_id):
    try:
        engine = engine_collection.find_one({ "_id": ObjectId(engine_id) })
        if not engine:
            raise EngineNotFound.EngineNotFoundException(engine_id)
    except EngineNotFound.EngineNotFoundException as e:
        print(e)
        return
    except Exception as e:
        print(f"Error al obtener el engine: {e}")
        return

    try:
        if engine['retrieval_model_id']:
            retrieval_config_db = config_collection.find_one(
                { "_id": ObjectId(engine['retrieval_model_id']) })
            if not retrieval_config_db:
                raise ModelNotFound.ModelNotFoundException(engine['retrieval_model_id'])
        else:
            raise ModelIdNotProvide.ModelIdNotProvideException("Retrieval")
        
        if engine['ranking_model_id']:
            ranking_config_db = config_collection.find_one(
                { "_id": ObjectId(engine['ranking_model_id']) })
            if not ranking_config_db:
                raise ModelNotFound.ModelNotFoundException(engine['ranking_model_id'])
        else:
            raise ModelIdNotProvide.ModelIdNotProvideException("Ranking")

    except ModelNotFound.ModelNotFoundException as e:
        print(f"Modelo no encontrado: {e}")
        return
    except ModelIdNotProvide.ModelIdNotProvideException as e:
        print(f"ID de modelo no proporcionado: {e}")
        return
    except Exception as e:
        print(f"Error inesperado: {e}")
        return

    print("***** Proceso de Fine Tunning Iniciado *****")
    engine_name = engine['name']
    is_training_by = engine["is_training_by"]
    service_models_path = f"service_models/{engine_name}"

    retrieval_config = ModelConfig(
        isTrain=retrieval_config_db['isTrain'],
        model_name=retrieval_config_db['name'],
        features=retrieval_config_db['features'],
        candidate_data_path=f"../../datasets/{retrieval_config_db['candidate_data_path']}.csv",
        data_path=f"../../datasets/{retrieval_config_db['data_path']}.csv",
        towers_layers_sizes=retrieval_config_db['towers_layers_sizes'],
        shuffle=retrieval_config_db['shuffle'],
        embedding_dimension=retrieval_config_db['embedding_dimension'],
        candidates_batch=retrieval_config_db['candidates_batch'],
        k_candidates=retrieval_config_db['k_candidates'],
        learning_rate=retrieval_config_db['learning_rate'],
        num_epochs=retrieval_config_db['num_epochs'],
        use_multiprocessing=retrieval_config_db['use_multiprocessing'],
        workers=retrieval_config_db['workers'],
        train_batch=retrieval_config_db['train_batch'],
        val_batch=retrieval_config_db['val_batch'],
        test_batch=retrieval_config_db['test_batch'],
        vocabularies_batch=retrieval_config_db['vocabularies_batch'],
        train_length=retrieval_config_db['train_length'],
        test_length=retrieval_config_db['test_length'],
        val_length=retrieval_config_db['val_length'],
        seed=retrieval_config_db['seed'],
        candidate_feature_merge=retrieval_config_db['candidate_feature_merge'],
        data_feature_merge=retrieval_config_db['data_feature_merge'],
        user_id_data=retrieval_config_db['user_id_data'],
        features_data_q=retrieval_config_db["features_data_q"],
        features_data_c=retrieval_config_db['features_data_c'],
        learning_rate_tunning=0.012,
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
        towers_layers_sizes=ranking_config_db['towers_layers_sizes'],
        deep_layers_sizes=ranking_config_db['deep_layers_sizes'],
        shuffle=ranking_config_db['shuffle'],
        embedding_dimension=ranking_config_db['embedding_dimension'],
        learning_rate=retrieval_config_db['learning_rate'],
        num_epochs=ranking_config_db['num_epochs'],
        use_multiprocessing=ranking_config_db['use_multiprocessing'],
        target_column={
            "current": ranking_config_db['target_column']['current'],
            "new": ranking_config_db['target_column']['new']
        },
        workers=ranking_config_db['workers'],
        train_batch=ranking_config_db['train_batch'],
        val_batch=ranking_config_db['val_batch'],
        test_batch=ranking_config_db['test_batch'],
        vocabularies_batch=ranking_config_db['vocabularies_batch'],
        train_length=ranking_config_db['train_length'],
        test_length=ranking_config_db['test_length'],
        val_length=ranking_config_db['val_length'],
        seed=ranking_config_db['seed'],
        candidate_feature_merge=ranking_config_db['candidate_feature_merge'],
        data_feature_merge=ranking_config_db['data_feature_merge'],
        user_id_data=ranking_config_db['user_id_data'],
        features_data_q=ranking_config_db["features_data_q"],
        features_data_c=ranking_config_db['features_data_c'],
        learning_rate_tunning=0.00001,
        to_map=True
    )


    if not is_training_by or retrieval_config.isTrain:
        # Reconstruye el modelo
        pipe = DataPipeline()
        try:
            candidate_df, = pipe.read_csv_data(paths=[
                retrieval_config.candidate_data_path
            ])
        except Exception as e:
            print(e)
            return

        try:
            candidate_df['descripcion'] = candidate_df['descripcion'].astype(str)
            candidate_df['nombre'] = candidate_df['nombre'].astype(str)
        except:
            print(f"Error al convertir 'descripcion' a string en pubs_df: {e}")
            print(f"Error al convertir 'nombre' a string en pubs_df: {e}")
            return


        try:
            pubs_ds = pipe.convert_to_tf_dataset(candidate_df)

            history_ds = pipe.load_dataset(retrieval_config_db['data_train_path'])
            vocabularies = pipe.load_vocabularies(path=retrieval_config_db['data_train_path'])

            total, train_Length, val_length, test_length = pipe.get_lengths(
                ds=history_ds,
                train_length=retrieval_config.train_length,
                test_length=retrieval_config.test_length,
                val_length=retrieval_config.val_length
            )

            train, val, test = pipe.split_into_train_and_test(
                ds=history_ds,
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
        except Exception as e:
            print(e)
            return
    
        # try:
        retrieval_model = retrieval_stage.model(
            config=retrieval_config,
            vocabularies=vocabularies,
            regularization_l2=0.05,
            candidates=pubs_ds,
        )

        retrieval_model.load_model(
            path=retrieval_config_db['modelPath'],
            cached_test=cached_test,
            cached_train=cached_train
        )
        # except Exception as e:
        #     print(e)
        #     return

        # Actualiza el modelo con los nuevos datos
        pipe = DataPipeline()
        try:
            history_df, = pipe.read_csv_data(paths=[
                f"../../datasets/{retrieval_config_db['data_tunning_name']}.csv"
            ])
            history_df = history_df.drop(['id'], axis=1)

            history_df = pipe.merge_data(
                left_data=candidate_df,
                right_data=history_df,
                left_on=retrieval_config.candidate_feature_merge,
                right_on=retrieval_config.data_feature_merge,
                output_features=retrieval_config.features
            )

            history_ds = pipe.convert_to_tf_dataset(history_df)

            vocabularies = pipe.build_vocabularies(
                features=retrieval_config.features,
                ds=history_ds,
                batch=retrieval_config.vocabularies_batch
            )

            total, train_length, val_length, test_length = pipe.get_lengths(
                ds=history_ds,
                train_length=retrieval_config.train_length,
                test_length=retrieval_config.test_length,
                val_length=retrieval_config.val_length
            )

            train, val, test = pipe.split_into_train_and_test(
                ds=history_ds,
                shuffle=retrieval_config.shuffle,
                train_length=train_length,
                val_length=val_length,
                test_length=test_length,
                seed=retrieval_config.seed
            )

            cached_train, cached_val, cached_test = pipe.data_caching(
                train=train,
                val=val,
                test=test,
                shuffle=retrieval_config.shuffle,
                train_batch=retrieval_config.train_batch_tunning,
                val_batch=retrieval_config.val_batch_tunning,
                test_batch=retrieval_config.test_batch_tunning
            )

            retrieval_model.learning_rate = retrieval_config.learning_rate_tunning
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
            retrieval_model.save_model(service_models_path, history_ds)
        except Exception as e:
            print(e)
            return 
        
        try:
            try:
                config_collection.update_many(
                    { '_id': retrieval_config_db['_id'] },
                    { '$set': {
                        "modelPath": retrieval_model.model_path,
                        "data_train_path": retrieval_model.data_train_path,
                        "metadata_path": retrieval_model.model_metadata_path,
                        "is_trainned": True,
                        "is_tunned": True
                    }}
                )
            except:
                raise UpdateDB.UpdateDBException(model_name="Retrieval")
        except UpdateDB.UpdateDBException as e:
            print(e)
            return e

        # ****************************************************************

        # Comprueba el modelo con todos los datos luego de ser actualizado
        # pipe = DataPipeline()
        # views_df, = pipe.read_csv_data(paths=[
        #     "../../datasets/vistas_no_nulas.csv"
        # ])
        # views_df = views_df[: shuffle + 10_000]
        # views_df = views_df.drop(['id'], axis=1)

        # views_df = pipe.merge_data(
        #     left_data=views_df,
        #     right_data=candidate_ds,
        #     left_on="publicacion_id",
        #     right_on="id",
        #     output_features=features
        # )

        # views_ds = pipe.convert_to_tf_dataset(views_df)

        # vocabularies = pipe.build_vocabularies(
        #     features=features,
        #     ds=views_ds,
        #     batch=1_000
        # )

        # total, train_Length, val_length, test_length = pipe.get_lengths(
        #     ds=views_ds,
        #     train_length=60,
        #     test_length=20,
        #     val_length=20
        # )

        # train, val, test = pipe.split_into_train_and_test(
        #     ds=views_ds,
        #     shuffle=shuffle,
        #     train_length=train_Length,
        #     val_length=val_length,
        #     test_length=test_length,
        #     seed=42
        # )

        # cached_train, cached_val, cached_test = pipe.data_caching(
        #     train=train,
        #     val=val,
        #     test=test,
        #     shuffle=shuffle,
        #     train_batch=train_batch,
        #     val_batch=val_batch,
        #     test_batch=test_batch
        # )

        # model.evaluate_model(
        #     cached_test=cached_test,
        #     cached_train=cached_train
        # )


    if not is_training_by or ranking_config.isTrain:
        # Reconstruye el modelo
        pipe = DataPipeline()
        try:
            history_ds = pipe.load_dataset(ranking_config_db['data_train_path'])
            vocabularies = pipe.load_vocabularies(
                path=ranking_config_db['data_train_path'])

            total, train_Length, val_length, test_length = pipe.get_lengths(
                ds=history_ds,
                train_length=ranking_config.train_length,
                test_length=ranking_config.test_length,
                val_length=ranking_config.val_length
            )

            train, val, test = pipe.split_into_train_and_test(
                ds=history_ds,
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
                regularization_l2=0.05,
                vocabularies=vocabularies,
            )

            ranking_model.load_model(
                path=ranking_config_db['modelPath'],
                cached_test=cached_test,
                cached_train=cached_train
            )

            # Actualiza el modelo con los nuevos datos
            pipe = DataPipeline()
            history_df, candidate_df = pipe.read_csv_data(paths=[
                f"../../datasets/{ranking_config_db['data_tunning_name']}.csv",
                ranking_config.candidate_data_path
            ])
            # history_df = history_df[ranking_config.shuffle: ranking_config.shuffle + 10_000]
            history_df = history_df.drop(['id'], axis=1)
            history_df[ranking_config.target_column['new']] = history_df[ranking_config.target_column['current']]\
                .map({True: 1, False: 0})
            
            all_features = ranking_config.features + [ranking_config.target_column['new']]
            history_df = pipe.merge_data(
                left_data=candidate_df,
                right_data=history_df,
                left_on=ranking_config.candidate_feature_merge,
                right_on=ranking_config.data_feature_merge,
                output_features=all_features
            )

            history_ds = pipe.convert_to_tf_dataset(history_df)

            vocabularies = pipe.build_vocabularies(
                features=ranking_config.features,
                ds=history_ds,
                batch=ranking_config.vocabularies_batch
            )

            total, train_Length, val_length, test_length = pipe.get_lengths(
                ds=history_ds,
                train_length=ranking_config.train_length,
                test_length=ranking_config.test_length,
                val_length=ranking_config.val_length
            )

            train, val, test = pipe.split_into_train_and_test(
                ds=history_ds,
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
                train_batch=ranking_config.train_batch_tunning,
                val_batch=ranking_config.val_batch_tunning,
                test_batch=ranking_config.test_batch_tunning
            )

            # Ajusta los hiperparámetros para fine-tuning
            ranking_model.learning_rate = ranking_config.learning_rate_tunning
            ranking_model.fit_model(
                cached_train=cached_train,
                cached_val=cached_val,

            )

            ranking_model.evaluate_model(
                cached_test=cached_test,
                cached_train=cached_train
            )
            ranking_model.save_model(service_models_path, history_ds)

        except Exception as e:
            print(e)
            return

        try:
            try:
                config_collection.update_many(
                    { '_id': ranking_config_db['_id'] },
                    { '$set': {
                        "modelPath": ranking_model.model_path,
                        "data_train_path": ranking_model.data_train_path,
                        "metadata_path": ranking_model.model_metadata_path,
                        "is_trainned": True,
                        "is_tunned": True
                    }}
                )
            except:
                raise UpdateDB.UpdateDBException(model_name="Retrieval")
        except UpdateDB.UpdateDBException as e:
            print(e)
            return e

if __name__ == "__main__":
    fine_tunning()