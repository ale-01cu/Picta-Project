import pandas as pd
import os
import logging
import matplotlib.pyplot as plt

# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dirname = os.path.dirname(__file__)

def user_frecuency():
    logging.info("Cargando datos de MovieLens y Picta Likes")
    movielens_ratings = pd.read_csv(os.path.join(dirname, "../datasets/movielens_ratings.csv"))
    picta_likes = pd.read_csv(os.path.join(dirname, "../datasets/likes.csv"))
    picta_views = pd.read_csv(os.path.join(dirname, "../datasets/vistas_no_nulas.csv"))
    picta_views = picta_views.sample(n=360_000, random_state=42)

    logging.info("Convirtiendo IDs de usuario a tipo string")
    movielens_ratings['user_id'] = movielens_ratings['user_id'].astype(str)
    picta_likes['usuario_id'] = picta_likes['usuario_id'].astype(str)

    freq_df1 = movielens_ratings['user_id'].value_counts()
    logging.info(f"Frecuencias en MovieLens:\n{freq_df1}")

    freq_df2 = picta_likes['usuario_id'].value_counts()
    logging.info(f"Frecuencias en Picta Likes:\n{freq_df2}")

    freq_df3 = picta_views['usuario_id'].value_counts()
    logging.info(f"Frecuencias en Picta Vistas:\n{freq_df3}")

    freq_df2_filtered = freq_df2[freq_df2 < 20]
    logging.info(f"Frecuencias filtradas en Picta Likes:\n{freq_df2_filtered}")

    freq_df3_filtered = freq_df3[freq_df3 < 20]
    logging.info(f"Frecuencias filtradas en Picta Vistas:\n{freq_df3_filtered}")

    ids_in_freq_df2 = freq_df2_filtered.index
    ids_in_freq_df3 = freq_df3_filtered.index
    # Asegurar que los IDs filtrados tengan valor True
    # freq_df2_filtered_without_dislikes = picta_likes.loc[picta_likes['usuario_id'].isin(ids_in_freq_df2)]
    # freq_df2_filtered_without_dislikes = freq_df2_filtered_without_dislikes[freq_df2_filtered_without_dislikes['valor'] == True]
    # freq_df2_filtered_without_dislikes = freq_df2_filtered_without_dislikes['usuario_id'].value_counts()
    # ids_in_freq_df2 = freq_df2_filtered_without_dislikes.index

    # freq_df2_filtered_without_dislikes = freq_df2_filtered_without_dislikes[freq_df2 < 20]
    # logging.info(f"Frecuencias filtradas en Picta Likes:\n{freq_df2_filtered_without_dislikes}")

    # filtered_picta_likes = picta_likes[
    #      ~picta_likes['usuario_id'].isin(ids_in_freq_df2)
    # ]
    filtered_picta_views = picta_views[~picta_views['usuario_id'].isin(ids_in_freq_df3)]
    logging.info(f"Filas filtradas en Picta vistas:\n{filtered_picta_views}")

    freq_df3 = filtered_picta_views['usuario_id'].value_counts()
    logging.info(f"Frecuencias en Picta Vistas:\n{freq_df3}")
    freq_df3_filtered = freq_df3[freq_df3 < 20]
    logging.info(f"Frecuencias filtradas en Picta Vistas:\n{freq_df3_filtered}")

    filtered_picta_views.to_csv("vistas_filtered.csv", index=False)

    # print(filtered_picta_views["valor"].value_counts())

    # comparison = pd.DataFrame({
    #     'movielens_frequency': freq_df1,
    #     'picta_likes_frequency': freq_df2
    # }).fillna(0)
    # logging.info(f"Comparación de frecuencias:\n{comparison}")

def comparasion():
    user_frecuency()

if __name__ == "__main__":
    comparasion()