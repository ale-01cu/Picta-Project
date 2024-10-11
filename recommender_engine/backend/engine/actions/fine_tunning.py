from engine.DataPipeline import DataPipeline
from stages.stages import retrieval_stage, ranking_Stage
from db.cruds.ModelCRUD import ModelCRUD
from db.cruds.EngineCRUD import EngineCRUD
from db.main import build_db
from db.config import engine
from models.ModelConfig import ModelConfig
from utils import read_json

def fine_tunning():

    print("***** Proceso de Fine Tunning Iniciado *****")
    engine_crud = EngineCRUD(engine=engine)
    engine_db = engine_crud.get_engine_running()
    engine_name = engine_db.name
    service_models_path = f"service_models/{engine_name}"

    model_crud = ModelCRUD(engine=engine)
    models = model_crud.get_models_running()
    retrieval_model_db = models[0]
    likes_model_db = models[1]

    retrieval_hiperparams = read_json(retrieval_model_db.metadata_path)
    likes_hiperparams = read_json(likes_model_db.metadata_path)

    print(retrieval_hiperparams)
 
    retrieval_config = ModelConfig(
        model_name=retrieval_hiperparams['model_name'],
        features=retrieval_hiperparams['features'],
        data_paths=["../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv"],
        towers_layers_sizes=retrieval_hiperparams['towers_layers_sizes'],
        shuffle=retrieval_hiperparams['shuffle'],
        embedding_dimension=retrieval_hiperparams['embedding_dimension'],
        candidates_batch=retrieval_hiperparams['candidates_batch'],
        k_candidates=retrieval_hiperparams['k_candidates'],
        learning_rate=retrieval_hiperparams['learning_rate'],
        num_epochs=retrieval_hiperparams['num_epochs'],
        use_multiprocessing=retrieval_hiperparams['use_multiprocessing'],
        workers=retrieval_hiperparams['workers'],
        train_batch=retrieval_hiperparams['train_batch'],
        val_batch=retrieval_hiperparams['val_batch'],
        test_batch=retrieval_hiperparams['test_batch'],
        vocabularies_batch=retrieval_hiperparams['vocabularies_batch'],
        train_length=retrieval_hiperparams['train_length'],
        test_length=retrieval_hiperparams['test_length'],
        val_length=retrieval_hiperparams['val_length'],
        seed=retrieval_hiperparams['seed'],
        features_data_q=retrieval_hiperparams["features_data_q"],
        features_data_c=retrieval_hiperparams['features_data_c'],
        to_map=True
    )
    


    # Likes Configs
    likes_config = ModelConfig(
        model_name=likes_hiperparams['model_name'],
        features=likes_hiperparams['features'],
        data_paths=likes_hiperparams['data_paths'],
        towers_layers_sizes=likes_hiperparams['towers_layers_sizes'],
        deep_layers_sizes=likes_hiperparams['deep_layers_sizes'],
        shuffle=likes_hiperparams['shuffle'],
        embedding_dimension=likes_hiperparams['embedding_dimension'],
        learning_rate=likes_hiperparams['learning_rate'],
        num_epochs=likes_hiperparams['num_epochs'],
        use_multiprocessing=likes_hiperparams['use_multiprocessing'],
        target_column={
            "current": likes_hiperparams['target_column']['current'],
            "new": likes_hiperparams['target_column']['new']
        },
        workers=likes_hiperparams['workers'],
        train_batch=likes_hiperparams['train_batch'],
        val_batch=likes_hiperparams['val_batch'],
        test_batch=likes_hiperparams['test_batch'],
        vocabularies_batch=likes_hiperparams['vocabularies_batch'],
        train_length=likes_hiperparams['train_length'],
        test_length=likes_hiperparams['test_length'],
        val_length=likes_hiperparams['val_length'],
        seed=likes_hiperparams['seed'],
        features_data_q=likes_hiperparams["features_data_q"],
        features_data_c=likes_hiperparams['features_data_c'],
        to_map=True
    )


    # Reconstruye el modelo
    pipe = DataPipeline()
    pubs_df, = pipe.read_csv_data(paths=retrieval_config.data_paths)

    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)

    pubs_ds = pipe.convert_to_tf_dataset(pubs_df)

    views_ds = pipe.load_dataset(retrieval_model_db.data_train_path)
    vocabularies = pipe.load_vocabularies(path=retrieval_model_db.data_train_path)

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


    model = retrieval_stage.retrieval_model(
        config=retrieval_config,
        vocabularies=vocabularies,
        candidates=pubs_ds,
    )

    model.load_model(
        path=retrieval_model_db.model_path,
        cached_test=cached_test,
        cached_train=cached_train
    )

    # Actualiza el modelo con los nuevos datos
    pipe = DataPipeline()
    views_df, = pipe.read_csv_data(paths=[
        "../../datasets/vistas_no_nulas.csv"
    ])
    views_df = views_df[retrieval_config.shuffle: retrieval_config.shuffle + 10_000]
    views_df = views_df.drop(['id'], axis=1)

    views_df = pipe.merge_data(
        left_data=views_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=retrieval_config.features
    )

    views_ds = pipe.convert_to_tf_dataset(views_df)

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

    model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,
        # learning_rate=learning_rate,
        # num_epochs=num_epochs,
        # use_multiprocessing=use_multiprocessing,
        # workers=workers
    )

    model.evaluate_model(
        cached_test=cached_test,
        cached_train=cached_train
    )
    model.save_model(service_models_path, views_ds)
    # Comprueba el modelo con todos los datos luego de ser actualizado
    # pipe = DataPipeline()
    # views_df, = pipe.read_csv_data(paths=[
    #     "../../datasets/vistas_no_nulas.csv"
    # ])
    # views_df = views_df[: shuffle + 10_000]
    # views_df = views_df.drop(['id'], axis=1)

    # views_df = pipe.merge_data(
    #     left_data=views_df,
    #     right_data=pubs_df,
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



    # Reconstruye el modelo
    pipe = DataPipeline()


    likes_ds = pipe.load_dataset(likes_model_db.data_train_path)
    vocabularies = pipe.load_vocabularies(
        path=likes_model_db.data_train_path)

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


    
    likes_model = ranking_Stage.likes_model(
        config=likes_config,
        vocabularies=vocabularies,
    )

    likes_model.load_model(
        path=likes_model_db.model_path,
        cached_test=cached_test,
        cached_train=cached_train
    )


    # Actualiza el modelo con los nuevos datos
    pipe = DataPipeline()
    likes_df, = pipe.read_csv_data(paths=[
        "../../datasets/likes.csv"
    ])
    likes_df = likes_df[likes_config.shuffle: likes_config.shuffle + 10_000]
    likes_df = likes_df.drop(['id'], axis=1)
    likes_df[likes_config.target_column['new']] = likes_df[likes_config.target_column['current']]\
        .map({True: 1, False: 0})

    likes_df = pipe.merge_data(
        left_data=likes_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=likes_config.features
    )

    likes_ds = pipe.convert_to_tf_dataset(likes_df)

    vocabularies = pipe.build_vocabularies(
        features=likes_config.features,
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

    # Ajusta los hiperparámetros para fine-tuning
    likes_model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,

    )

    likes_model.evaluate_model(
        cached_test=cached_test,
        cached_train=cached_train
    )
    likes_model.save_model(service_models_path, likes_ds)


    model_crud.turn_off_all()

    model_crud.create(
        name=model.model_name,
        stage="retrieval",
        model_path=model.model_path,
        data_train_path=model.data_train_path,
        metadata_path=model.model_metadata_path,
        engine_id=engine_db.id
    )

    model_crud.create(
        name=likes_model.model_name,
        stage="ranking",
        model_path=likes_model.model_path,
        data_train_path=likes_model.data_train_path,
        metadata_path=likes_model.model_metadata_path,
        engine_id=engine_db.id
    )

    engine_crud.close_session()
    model_crud.close_session()


if __name__ == "__main__":
    fine_tunning()