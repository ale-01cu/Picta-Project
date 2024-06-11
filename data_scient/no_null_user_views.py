import pandas as pd
import time


chunksize = 5 * 10**6
counter = 1

start = time.time()

dataset_path = 'C:/Users/Picta/Desktop/Picta-Project/datasets/visitas.csv'
dataset_path = 'I:/UCI/tesis/Picta-Project/datasets/visitas.csv'

all_null_rows_df = pd.DataFrame()


for chunk in pd.read_csv(dataset_path, chunksize=chunksize):
    print('Numero: ', counter, 'Chunk: ', len(chunk))
    counter += 1
    nulas_en_X_chunk = chunk[~chunk['usuario_id'].isnull()]
    
    all_null_rows_df = pd.concat(
        [all_null_rows_df, nulas_en_X_chunk], 
        ignore_index=True
    )

print(len(all_null_rows_df))
all_null_rows_df.to_csv('vistas_no_nulas.csv', index=False)

# print(df.head())
# print('----------')
# print(df.info())
# print('----------')
# print(df.describe())

end = time.time()

print(round(end - start, 2))