from DataPipeline import DataPipeline
import FeaturesTypes
from stages.RetrievalStage import RetrievalStage


def train():
    print("***** Proceso de Entrenamiendo Iniciado *****")
 
    features = ['usuario_id', 'id']
    shuffle = 100_000
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
 
    pipe = DataPipeline()
    pubs_df, views_df = pipe.read_csv_data(paths=[
        "../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv",
        "../../datasets/vistas_no_nulas.csv"
    ])

    views_df = views_df[: shuffle]
    views_df = views_df.drop(['id'], axis=1)
    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)


    views_df = pipe.merge_data(
        left_data=views_df,
        right_data=pubs_df,
        left_on="publicacion_id",
        right_on="id",
        output_features=features
    )

    pubs_ds = pipe.convert_to_tf_dataset(pubs_df)
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

    pipe.close()

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

    model.fit_model(
        cached_train=cached_train,
        cached_val=cached_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_multiprocessing=use_multiprocessing,
        workers=workers   
    )
    model.evaluate_model(
        cached_test=cached_test,
        cached_train=cached_train
    )
    model.save_model("service_models")


if __name__ == "__main__":
    train()