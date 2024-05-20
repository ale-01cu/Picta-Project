from .RatingWithTimestamp import RatingWithTimestamp
from .ItemToItem import ItemToItem
from .UserClicksHIstory import UserClicksHistory
from .SecuentialItems import SecuentialItems
from .LikesWithTimestamp import LikesWithTimestamp
from .PostiveFeatures import PositiveFeaturesGenerator

FROM_DATASET_PATH = './datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
TO_DATASET_PATH = './datasets/publicaciones_ratings_con_timestamp_medium.csv'

TO_DATASET_ITEM_TO_ITEM_PATH = './datasets/publicacion_a_publicacion_con_timestamp.csv'
TO_DATASET_USER_CLICKS_HISTORY_PATH = './datasets/historial_clicks_usuario.csv'

TO_DATASET_SEQUENCE_ITEMS_PATH = './datasets/historial_secuencia_publicaciones.csv'

TO_DATASET_LIKES_PATH = './datasets/likes_con_timestamp_100K.csv'

TO_DATASET_PF_PATH = './datasets/positive_features_with_timestamp_1m.csv'

def generate_ratings_with_timestamp():
    gn = RatingWithTimestamp(
        from_path=FROM_DATASET_PATH, 
        to_path=TO_DATASET_PATH
    )
    gn()


def generate_item_to_item_dataset():
    iti = ItemToItem(
        from_path=FROM_DATASET_PATH, 
        to_path=TO_DATASET_ITEM_TO_ITEM_PATH,
        num_rows=100_000
    )
    iti()


def generate_user_clicks_history():
    uch = UserClicksHistory(
        from_path=FROM_DATASET_PATH,
        to_path=TO_DATASET_USER_CLICKS_HISTORY_PATH,
        num_rows=100_000,
        users_ids_range=50
    )
    uch()


def generate_candidate_sequence():
    si = SecuentialItems(
        from_path=FROM_DATASET_PATH,
        to_path=TO_DATASET_SEQUENCE_ITEMS_PATH,
        num_rows=1_000_000,
    )
    si(k_sequence=10)


def generate_likes_with_timestamp():
    lt = LikesWithTimestamp(
        from_path=FROM_DATASET_PATH,
        to_path=TO_DATASET_LIKES_PATH,
        num_rows=100_000,
        users_ids_range=23487,
        seed=11
    )

    lt()

def generate_positive_features_with_timestamp():
    pf = PositiveFeaturesGenerator(
        from_path=FROM_DATASET_PATH,
        to_path=TO_DATASET_PF_PATH,
        num_rows=1_000_000,
        users_ids_range=50_000,
        seed=25
    )

    pf(categories=['like', 'download', 'comment'])