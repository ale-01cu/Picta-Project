import pandas as pd
import random
import time
from datetime import datetime

FROM_DATASET_PATH = './datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
TO_DATASET_PATH = './datasets/publicaciones_ratings_con_timestamp_medium.csv'

def generate_users_id(num_users, users_ids_range):
    return [
        random.randint(1, users_ids_range) 
        for _ in range(num_users)
    ]

def choice_publications_ids(id_list, amount):
    return random.choices(id_list, k=amount)

def generate_ratings(num_rows):
    ratings = [random.randint(10, 50) / 10.0 for _ in range(num_rows)]
    return ratings

def delete_duplicates(df):
    df_temp = df.drop_duplicates(subset=['user_id', 'publication_id'], keep='first')
    df_temp = df.drop_duplicates(subset=['user_id', 'timestamp'], keep='first')
    return df_temp


def generate_dates(
    num_rows: int,
    start: datetime = time.mktime(time.strptime('01-01-2020', '%d-%m-%Y')), 
    end: datetime = time.mktime(time.strptime('31-12-2024', '%d-%m-%Y'))):
   
    return [
        random.randint(start, end)
        for _ in range(num_rows)
    ]


def generate_table(num_rows, users_ids_range, publications_ids, to_dataset_path):
    # Genera un 50% más de filas de las que necesitas para tener suficientes después de eliminar las duplicadas
    num_rows_generated = int(num_rows * 1.5)
    users_ids = generate_users_id(num_rows_generated, users_ids_range)
    publications_ids = choice_publications_ids(publications_ids, num_rows_generated)
    ratings = generate_ratings(num_rows_generated)
    timestamps = generate_dates(num_rows_generated)

    df = pd.DataFrame({
        'user_id': users_ids, 
        'publication_id': publications_ids, 
        'rating': ratings,
        'timestamp': timestamps
    })

    df = df.sort_values('timestamp', ascending=False)

    df = delete_duplicates(df.copy())
    # Recorta el DataFrame a la cantidad de filas que quieres
    df = df.iloc[:num_rows]
    df.to_csv(to_dataset_path, index=False)




def main():
    num_rows = 100_000
    users_ids_range = 50
    df = pd.read_csv(FROM_DATASET_PATH)
    publications_ids = df['id'].unique().tolist()
    generate_table(num_rows, users_ids_range, publications_ids, TO_DATASET_PATH)

if __name__ == '__main__':
    main()