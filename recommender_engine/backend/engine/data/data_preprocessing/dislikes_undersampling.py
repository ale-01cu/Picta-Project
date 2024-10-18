import pandas as pd

def dislikes_undersampling(df) -> pd.DataFrame:
    df['fecha'] = pd.to_datetime(df['fecha'])

    df = df.sort_values(by='fecha')

    # Elimina las filas más antiguas de dislikes
    num_filas_eliminar = 370_000  # número de filas a eliminar
    dislikes_viejos = df[(df['valor'] == 1)].head(num_filas_eliminar)

    # Elimina las filas de dislikes más antiguas del dataset
    df.drop(dislikes_viejos.index, inplace=True)

    # Verifica la nueva distribución de likes y dislikes
    likes = df['valor'].value_counts()[1]
    dislikes = df['valor'].value_counts()[0]

    print(f"Likes: {likes}")
    print(f"Dislikes: {dislikes}")
    print(f"Proporción de likes: {likes / (likes + dislikes):.2f}")
    print(f"Proporción de dislikes: {dislikes / (likes + dislikes):.2f}")

    return df
    

if "__main__" == __name__:
    dislikes_undersampling()