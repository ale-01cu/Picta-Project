from DataPipeline import DataPipeline
import FeaturesTypes
from stages.stages import retrieval_stage, ranking_Stage
from db.cruds.EngineCRUD import EngineCRUD
from db.cruds.ModelCRUD import ModelCRUD
from db.config import engine
from db.main import build_db
build_db()

def train():
    global engine

    # General Configs
    engine_name = "Engine_v0.2"
    service_models_path = f"service_models/{engine_name}"

    # Retrieval Configs
    retrieval_model_name = "Retrieval lite"
    retrieval_features = ['usuario_id', 'id']
    shuffle = 10_000
    embedding_dimension = 64
    candidates_batch = 128
    k_candidates = 100
    learning_rate = 0.1
    num_epochs = 1
    use_multiprocessing = True
    workers = 4
    train_batch = 8192
    val_batch = 4096
    test_batch = 4096

    # Likes Configs
    ranking_features = ['usuario_id', 'id', "like_dislike"]
    likes_model_name = "Likes lite"

    
    print(f"***** Proceso de Entrenamiendo del Systema {engine_name} Iniciado *****")
 
    pipe = DataPipeline()
    pubs_df, views_df = pipe.read_csv_data(paths=[
        "../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv",
        "../../datasets/vistas_no_nulas.csv"
    ])

    pubs_df = pubs_df[: 5000]

    views_df = views_df[: shuffle]
    views_df = views_df.drop(['id'], axis=1)
    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)


    views_df = pipe.merge_data(
        left_data=views_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=retrieval_features
    )

    pubs_ds = pipe.convert_to_tf_dataset(pubs_df)
    views_ds = pipe.convert_to_tf_dataset(views_df)

    vocabularies = pipe.build_vocabularies(
        features=retrieval_features,
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

    pipe.close()

    retrieval_model = retrieval_stage.retrieval_model(
        model_name=retrieval_model_name,
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

    retrieval_model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers   
    )
    retrieval_model.evaluate_model(
        cached_test=cached_test,
        cached_train=cached_train
    )
    retrieval_model.save_model(
        service_models_path, views_ds)


    pipe = DataPipeline()
    likes_df, = pipe.read_csv_data(paths=[
        "../../datasets/likes.csv"
    ])
    likes_df = likes_df.drop(['id'], axis=1)
    likes_df['like_dislike'] = likes_df['valor']\
        .map({True: 1, False: 0})

    likes_df = pipe.merge_data(
        left_data=likes_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=ranking_features
    )

    likes_ds = pipe.convert_to_tf_dataset(likes_df)
    
    vocabularies = pipe.build_vocabularies(
        features=ranking_features,
        ds=likes_ds,
        batch=1_000
    )

    total, train_Length, val_length, test_length = pipe.get_lengths(
        ds=likes_ds,
        train_length=60,
        test_length=20,
        val_length=20
    )

    train, val, test = pipe.split_into_train_and_test(
        ds=likes_ds,
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

    likes_model = ranking_Stage.likes_model(
        model_name=likes_model_name,
        towers_layers_sizes=[],
        deep_layers_sizes=[],
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
        embedding_dimension=64, 
    )


    likes_model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers   
    )

    likes_model.evaluate_model(
        cached_test=cached_test,
        cached_train=cached_train
    )

    likes_model.save_model(service_models_path, likes_ds)

    engine_crud = EngineCRUD(engine=engine)
    engine_crud.turn_off_all()
    new_engine = engine_crud.create(name=engine_name)
    engine_crud.close_session()

    model_crud = ModelCRUD(engine=engine)
    model_crud.turn_off_all()

    model_crud.create(
        name=retrieval_model.model_name,
        stage="retrieval",
        model_path=retrieval_model.model_path,
        data_train_path=retrieval_model.model_path,
        metadata_path=retrieval_model.model_metadata_path,
        engine_id=new_engine.id
    )


    model_crud.create(
        name=likes_model.model_name,
        stage="ranking",
        model_path=likes_model.model_path,
        data_train_path=likes_model.model_path,
        metadata_path=likes_model.model_metadata_path,
        engine_id=new_engine.id
    )

    model_crud.close_session()


if __name__ == "__main__":
    train()