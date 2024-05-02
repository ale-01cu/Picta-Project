from .RatingWithTimestamp import RatingWithTimestamp
from .ItemToItem import ItemToItem
from .UserClicksHIstory import UserClicksHistory

FROM_DATASET_PATH = './datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
TO_DATASET_PATH = './datasets/publicaciones_ratings_con_timestamp_medium.csv'

TO_DATASET_ITEM_TO_ITEM_PATH = './datasets/publicacion_a_publicacion_con_timestamp.csv'
TO_DATASET_USER_CLICKS_HISTORY_PATH = './datasets/historial_clicks_usuario.csv'

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