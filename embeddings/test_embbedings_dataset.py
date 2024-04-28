import pandas as pd
import ast

df = pd.read_csv('../datasets/picta_publicaciones_procesadas_sin_nulas_embeddings.csv')
first_row = df.head(1)
name_vector_str = first_row['descripcion'].values
name_vector = list(ast.literal_eval(name_vector_str[0]))


for n, s in zip(name_vector, name_vector_str[0].split(', ')):
    if(str(n) != s):
        print(n, s)