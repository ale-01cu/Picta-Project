from engine.data.DataPipeline import DataPipeline
from engine.data.data_preprocessing.transform_date_to_timestamp import transform_date_to_timestamp
from engine.data.data_preprocessing.dislikes_undersampling import dislikes_undersampling
from engine.data.data_preprocessing.add_age_column import add_age_column
import pandas as pd

def data_preprocessing():
    pipe = DataPipeline()
    # likes_df, views_df = pipe.read_csv_data(paths=[
    #     "../../../../datasets/likes.csv",
    #     "../../../../datasets/vistas_no_nulas.csv"
    # ])

    # likes_df = dislikes_undersampling(likes_df)
    # likes_df = transform_date_to_timestamp(likes_df, "fecha")

    # views_df = transform_date_to_timestamp(views_df, 'fecha')
    # user_df = pd.DataFrame()
    # user_df['usuario_id'] = pd.concat([likes_df['usuario_id'], views_df['usuario_id']]).reset_index(drop=True)
    # user_df['usuario_id'] = user_df['usuario_id'].drop_duplicates().reset_index(drop=True)
    # user_df = user_df.dropna(subset=['usuario_id'])

    # user_df['usuario_id'] = user_df['usuario_id'].astype(int)
    # user_df = add_age_column(user_df, 'edad')
    
    # likes_df = likes_df.merge(user_df, on='usuario_id')
    # views_df = views_df.merge(user_df, on='usuario_id')

    # user_df.to_csv('datasets/usuarios.csv', index=False)
    # likes_df.to_csv('datasets/likes.csv', index=False)
    # views_df.to_csv('datasets/vistas.csv', index=False)


    users_df, = pipe.read_csv_data(paths=[
        "../../datasets/usuarios.csv"
    ])

    users_df = transform_date_to_timestamp(users_df, "fecha_nacimiento")
    users_df.to_csv("datasets/usuarios_timestamp.csv", index=False)

    

if "__main__" == __name__:
    data_preprocessing()