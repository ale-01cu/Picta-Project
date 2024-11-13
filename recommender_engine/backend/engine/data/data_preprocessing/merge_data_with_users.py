import pandas as pd

def merge_data_with_users(likes_path, usuarios_path, output_path):
    # Cargar los datasets
    likes = pd.read_csv(likes_path)
    usuarios = pd.read_csv(usuarios_path)
    
    # Eliminar duplicados en el dataset de usuarios basado en el campo 'id'
    usuarios = usuarios.drop_duplicates(subset='id')
    
    # Unir los datasets en los campos 'usuario_id' y 'id'
    likes_completados = pd.merge(likes, usuarios, left_on='usuario_id', right_on='id', how='left')
    
    # Guardar el dataset completado en un nuevo archivo
    likes_completados.to_csv(output_path, index=False)


if __name__ == "__main__":
    merge_data_with_users('../../../datasets/vistas_filtered.csv', '../../../datasets/usuarios_timestamp.csv', '../../../datasets/vistas.csv')
    # merge_data_with_users('./likes.csv', './usuarios_timestamp.csv', './likes_full.csv')
