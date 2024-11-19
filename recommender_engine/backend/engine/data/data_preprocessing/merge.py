import pandas as pd

def completar_likes_con_usuarios(data_df, users_df, output_path):
    # Cargar los datasets
    likes = data_df.copy()
    usuarios = users_df.copy()

    # Eliminar duplicados en el dataset de usuarios basado en el campo 'id'
    usuarios = usuarios.drop_duplicates(subset='id')
    
    # Unir los datasets en los campos 'usuario_id' y 'id'
    likes_completados = pd.merge(likes, usuarios, left_on='usuario_id', right_on='id', how='left')
    
    # Guardar el dataset completado en un nuevo archivo
    likes_completados.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Ejemplo de uso
    completar_likes_con_usuarios('./vistas.csv', './usuarios_timestamp.csv', './vistas.csv')
    # completar_likes_con_usuarios('./likes.csv', './usuarios_timestamp.csv', './likes_full.csv')

