from DataPipeline import DataPipeline
from models.RetrievalModel import RetrievalModel
import FeaturesTypes
from stages.RetrievalStage import RetrievalStage
import pickle
from db.cruds.ModelCRUD import ModelCRUD
from db.cruds.EngineCRUD import EngineCRUD
from db.main import build_db
from db.config import engine
import tensorflow as tf
import os

build_db()
def fine_tunning():

    print("***** Proceso de Fine Tunning Iniciado *****")
    engine_db = EngineCRUD(engine=engine).get_engine_running()
    engine_name = engine_db.name
    service_models_path = f"service_models/{engine_name}"
 
    features = ['usuario_id', 'id']
    shuffle = 100_000
    embedding_dimension = 64
    candidates_batch = 128
    k_candidates = 100
    learning_rate = 0.001
    num_epochs = 1
    use_multiprocessing = True
    workers = 4
    train_batch = 8192
    val_batch = 4096
    test_batch = 4096
    
    model_crud = ModelCRUD(engine=engine)
    models = model_crud.get_models_running()
    model_db = models[0]

    # Reconstruye el modelo
    pipe = DataPipeline()
    pubs_df, = pipe.read_csv_data(paths=[
        "../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv",
    ])

    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)

    pubs_ds = pipe.convert_to_tf_dataset(pubs_df)

    views_ds = pipe.load_dataset(model_db.data_train_path)
    vocabularies = pipe.load_vocabularies(path=model_db.data_train_path)

    total, train_Length, val_length, test_length = pipe.get_lengths(
        ds=views_ds,
        train_length=60,
        test_length=20,
        val_length=20
    )

    train, val, test = pipe.split_into_train_and_test(
        ds=views_ds,
        shuffle=shuffle,
        train_length=train_Length,
        val_length=val_length,
        test_length=test_length,
        seed=42
    )

    cached_train, cached_val, cached_test = pipe.data_caching(
        train=train,
        val=val,
        test=test,
        shuffle=shuffle,
        train_batch=train_batch,
        val_batch=val_batch,
        test_batch=test_batch
    )


    retrieval_stage = RetrievalStage()
    model = retrieval_stage.retrieval_model(
        model_name="Retrieval Lite",
        towers_layers_sizes=[],
        vocabularies=vocabularies,
        features_data_q={
            'usuario_id': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3 }    
        },
        features_data_c={ 
            'id': { 'dtype': FeaturesTypes.CategoricalInteger, 'w': 1 },
            # 'nombre': { 'dtype': StringText.StringText, 'w': 0.2 },
            # 'descripcion': { 'dtype': StringText.StringText, 'w': 0.1 }
        },
        embedding_dimension=embedding_dimension, 
        # test=test, 
        # shuffle=10_000, 
        # test_batch=512, 
        candidates=pubs_ds,
        candidates_batch=candidates_batch, 
        k_candidates=k_candidates
    )

    model.load_model(
        path=model_db.model_path,
        cached_test=cached_test,
        cached_train=cached_train
    )

    # Actualiza el modelo con los nuevos datos
    pipe = DataPipeline()
    views_df, = pipe.read_csv_data(paths=[
        "../../datasets/vistas_no_nulas.csv"
    ])
    views_df = views_df[shuffle: shuffle + 10_000]
    views_df = views_df.drop(['id'], axis=1)

    views_df = pipe.merge_data(
        left_data=views_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=features
    )

    views_ds = pipe.convert_to_tf_dataset(views_df)

    vocabularies = pipe.build_vocabularies(
        features=features,
        ds=views_ds,
        batch=1_000
    )

    total, train_Length, val_length, test_length = pipe.get_lengths(
        ds=views_ds,
        train_length=60,
        test_length=20,
        val_length=20
    )

    train, val, test = pipe.split_into_train_and_test(
        ds=views_ds,
        shuffle=shuffle,
        train_length=train_Length,
        val_length=val_length,
        test_length=test_length,
        seed=42
    )

    cached_train, cached_val, cached_test = pipe.data_caching(
        train=train,
        val=val,
        test=test,
        shuffle=shuffle,
        train_batch=train_batch,
        val_batch=val_batch,
        test_batch=test_batch
    )

    model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers
    )

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

    model.save_model(service_models_path, views_ds)


    model_crud.turn_off_all()

    model_crud.create(
        name=model.model_name,
        stage="retrieval",
        model_path=model.model_path,
        data_train_path=model.model_path,
        metadata_path=model.model_metadata_path,
        engine_id=engine_db.id
    )


if __name__ == "__main__":
    fine_tunning()