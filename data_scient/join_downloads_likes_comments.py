import pandas as pd

BASE_URL = 'I:/UCI/tesis/Picta-Project/datasets/'
LIKES_URL = f'{BASE_URL}likes.csv'
DOWNLOADS_URL = f'{BASE_URL}descargas.csv'
COMMENTS_URL = f'{BASE_URL}comentarios.csv'

likes_df = pd.read_csv(LIKES_URL)
downloads_df = pd.read_csv(DOWNLOADS_URL)
comments_df = pd.read_csv(COMMENTS_URL)

print(likes_df.columns)
print(downloads_df.columns)
print(comments_df.columns)


print(likes_df.info())
print(downloads_df.info())
print(comments_df.info())


columns = ['usuario_id', 'publicacion_id', 'categoria']
likes_columns = ['usuario_id', 'publicacion_id', 'valor']
downloads_columns = ['usuario_id', 'publicacion_id']
comments_columns = ['usuario_id', 'publicacion_id', 'publicado', 'eliminado']

likes_df = likes_df[likes_columns].loc[likes_df['valor']]
downloads_df = downloads_df[downloads_columns]
comments_df = comments_df[comments_columns].loc[(comments_df['publicado']) & (~comments_df['eliminado'])]

print('Likes', likes_df.shape)
print('Descargas', downloads_df.shape)
print('Comentarios', comments_df.shape)

likes_df['categoria'] = 'like'
downloads_df['categoria'] = 'download'
comments_df['categoria'] = 'comment'

likes_df = likes_df[columns]
downloads_df = downloads_df[columns]
comments_df = comments_df[columns]


df = pd.concat([likes_df, downloads_df, comments_df], axis=0)
df = df.sample(frac=1)

print(df.shape)

print('Salvando la data...')
df.to_csv('../datasets/positive_data.csv')


