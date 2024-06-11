import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

models = ['all-MiniLM-L6-v2', 'sentence-transformers/stsb-roberta-large', 'all-mpnet-base-v2']
model = SentenceTransformer(models[0])

df = pd.read_csv('C:/Users/Picta/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')

df = df.drop(['id', 'Unnamed: 0', 'categoria'], axis=1)
print(df.columns)
output_file = '../datasets/pubs_embeddings.npy'
pubs_matrix = None

pca = PCA(n_components=3)

# with open(output_file, 'wb') as f:
for index, row in df.iterrows():
    embeddings = model.encode(list(dict(row).values()))
    if pubs_matrix is None:
        pubs_matrix = np.concatenate(embeddings)
        # pubs_reduced = PCA(n_components=2).fit_transform([pubs_matrix])

    else:
    # print(embeddings)
        pubs_matrix = np.vstack((pubs_matrix, np.concatenate(embeddings)))
    
    print(f'Índice: {index}') 

print(pubs_matrix)
pubs_reduced = pca.fit_transform(pubs_matrix)
np.save(output_file, pubs_reduced)