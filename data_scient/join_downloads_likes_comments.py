import pandas as pd
from dateutil import parser 

def convert_to_timestamp(date_string):
    date_object = parser.parse(date_string)
    return date_object.timestamp()


BASE_URL = 'I:/UCI/tesis/Picta-Project/datasets/'
LIKES_URL = f'{BASE_URL}likes.csv'
DOWNLOADS_URL = f'{BASE_URL}descargas.csv'
COMMENTS_URL = f'{BASE_URL}comentarios.csv'

print('Cargando la data...')
likes_df = pd.read_csv(LIKES_URL)
downloads_df = pd.read_csv(DOWNLOADS_URL)
comments_df = pd.read_csv(COMMENTS_URL)

print(likes_df.columns)
print(downloads_df.columns)
print(comments_df.columns)


print(likes_df.info())
print(downloads_df.info())
print(comments_df.info())


columns = ['usuario_id', 'publicacion_id', 'category', 'fecha']
likes_columns = ['usuario_id', 'publicacion_id', 'valor', 'fecha']
downloads_columns = ['usuario_id', 'publicacion_id', 'fecha']
comments_columns = ['usuario_id', 'publicacion_id', 'publicado', 'eliminado', 'fecha']

print('Filtrando la data...')
likes_df = likes_df[likes_columns].loc[likes_df['valor']]
downloads_df = downloads_df[downloads_columns]
comments_df = comments_df[comments_columns].loc[(comments_df['publicado']) & (~comments_df['eliminado'])]

print('Likes', likes_df.shape)
print('Descargas', downloads_df.shape)
print('Comentarios', comments_df.shape)

print('Procesando las fechas...')
print('likes...')
likes_df['fecha'] = likes_df['fecha'].apply(convert_to_timestamp)
print('descargas...')
downloads_df['fecha'] = downloads_df['fecha'].apply(convert_to_timestamp)
print('comentarios...')
comments_df['fecha'] = comments_df['fecha'].apply(convert_to_timestamp)

likes_df['category'] = 'like'
downloads_df['category'] = 'download'
comments_df['category'] = 'comment'

likes_df = likes_df[columns]
downloads_df = downloads_df[columns]
comments_df = comments_df[columns]


print('Uniendo la data...')
df = pd.concat([likes_df, downloads_df, comments_df], axis=0)
df = df.sample(frac=1)

print(df.shape)

print('Salvando la data...')
df.to_csv('../datasets/positive_data.csv')





def analysing_positive_data():
    positive_df = pd.read_csv('../datasets/positive_data.csv')
    nulas_en_X_chunk = positive_df[positive_df['usuario_id'].isnull()]
    print(len(nulas_en_X_chunk))
