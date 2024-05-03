from .RetrievalModel import RetrievalModel
from .RankingModel import RankingModel
from .ListWiseRankingModel import ListWiseRankingModel
from .ItemToItemRetrievalModel import ItemToItemRetrievalModel
from .SecuentialRetrievalModel import SecuntialRetrievalModel
from .DCN_Ranking import DCN
import time
from .data_pipeline import train, test, pubs_ds
import tensorflow_ranking as tfr
import tensorflow as tf
from .data.DataPipelineItemToItem import DataPipelineItemToItem
from .data.DataPipelineSecuential import DataPipelineSecuential
import pandas as pd

def use_retrieval_model(user_id):

    model = RetrievalModel(layer_sizes=[32], embedding_dimension=32, 
        train=train, test=test, shuffle=100_000, train_batch=8192, test_batch=1024, 
        candidates=pubs_ds, candidates_batch=128, k_candidates=100
    )

    try:
        loaded = model.load('./recommender_engine/models/test-models')
        return loaded({'user_id': [user_id]})

    except Exception as e:
        model.fit_model(learning_rate=0.1, num_epochs=8)
        ids = model.predict_model(user_id=user_id)
        model.save('./recommender_engine/models/test-model')
        return ids


def use_ranking_model(user_id, ids):
    model = RankingModel(layer_sizes=[32], deep_layer_sizes=[512, 512, 512],
        train=train, test=test, shuffle=100_000, train_batch=8192, test_batch=4096,
        embedding_dimension=32
    )
    model.fit_model(learning_rate=0.1, num_epochs=30)
    model.evaluate_model()
    # model.predict_model(user_id, ids)


def use_dcn_ranking_model(user_id, ids):
    EMBEDDING_DIMENSION=32
    DEEP_LAYER_SIZES_RANKING = [512, 512, 512]
    DEEP_LAYER_SIZES_TOWERS = [32]
    USE_CROSS_LAYER = True
    PROJECTION_DIM = 32
    SHUFFLE=100_000
    TRAIN_BATCH=8192
    TEST_BATCH=4096
    LEARNING_RATE = 0.1
    EPOCHS = 12

    model = DCN(
        deep_layer_sizes_ranking=DEEP_LAYER_SIZES_RANKING, 
        deep_layer_sizes_towers=DEEP_LAYER_SIZES_TOWERS,
        train=train,
        test=test,
        shuffle=SHUFFLE,
        train_batch=TRAIN_BATCH,
        test_batch=TEST_BATCH,
        embedding_dimension=EMBEDDING_DIMENSION,
        use_cross_layer=USE_CROSS_LAYER, 
        projection_dim=PROJECTION_DIM,
    )
    model.fit_model(learning_rate=LEARNING_RATE, num_epochs=EPOCHS)
    model.evaluate_model()


def use_listwise_ranking_model():
    losses = [
        tf.keras.losses.MeanSquaredError(),
        tfr.keras.losses.PairwiseHingeLoss(),
        tfr.keras.losses.ListMLELoss() # En teoria esta es la mejor
    ]


    model = ListWiseRankingModel(
        loss=losses[2],
        layer_sizes=[32], deep_layer_sizes=[512],
        train=train, test=test, shuffle=100_000, train_batch=256, 
        test_batch=12, embedding_dimension=32
    )
    model.fit_model(learning_rate=0.1, num_epochs=30)
    model.evaluate_model()



def use_item_to_item_model():
    pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)
    pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))


    path = 'C:/Users/Ale/Desktop/Picta-Project/datasets/publicacion_a_publicacion_con_timestamp.csv'
    dp = DataPipelineItemToItem(dataframe_path=path)
    train, test, vocabularies = dp(df_to_merge=pubs_df)

    model = ItemToItemRetrievalModel(
        layer_sizes=[32], embedding_dimension=32, 
        train=train, test=test, shuffle=100_000, train_batch=8192, test_batch=4096, 
        candidates=pubs_ds, candidates_batch=128, k_candidates=100,
        vocabularies=vocabularies, 
        features_names_q=['publication_id_q', 'nombre_q'],
        features_names_c=['id', 'nombre'], 
    )
    model.fit_model(learning_rate=0.1, num_epochs=3)
    model.evaluate_model()


def use_secuential_model():
    pubs_df = pd.read_csv('C:/Users/Ale/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)
    pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))


    path = 'C:/Users/Ale/Desktop/Picta-Project/datasets/historial_secuencia_publicaciones.csv'
    dp = DataPipelineSecuential(dataframe_path=path)
    train, test, vocabularies = dp(df_to_merge=pubs_df)

    # model = SecuntialRetrievalModel(
    #     layer_sizes=[34], train=train, test=test,
    #     candidates=pubs_ds, embedding_dimension=32, 
    #     shuffle=100_000, train_batch=8192, test_batch=4096,
    #     candidates_batch=128, k_candidates=100
    # )

    # model.fit_model(learning_rate=0.1, num_epochs=3)
    # model.evaluate_model()


if __name__ == "__main__":
    USER_ID = '26'

    start = time.time()

    # score, ids = use_retrieval_model(USER_ID)
    # use_ranking_model(USER_ID, [])
    # use_dcn_ranking_model(USER_ID, [])
    # use_listwise_ranking_model()
    # use_item_to_item_model()
    use_secuential_model()

    end = time.time()

    print(round(end - start, 2))

