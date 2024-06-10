import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA

models = ['all-MiniLM-L6-v2', 'sentence-transformers/stsb-roberta-large', 'all-mpnet-base-v2']
model = SentenceTransformer(models[0])

df = pd.read_csv('C:/Users/Picta/Desktop/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')


output_file = 'pubs_embeddings.npy'
pubs_matrix = None

pca = PCA(n_components=2)

# with open(output_file, 'wb') as f:
for index, row in df.iterrows():
    embeddings = model.encode(list(dict(row).values()))
    if pubs_matrix is None:
        pubs_matrix = np.concatenate(embeddings)
        pubs_reduced = PCA(n_components=1).fit_transform([pubs_matrix])

    else:
        pubs_matrix = np.vstack((pubs_matrix, np.concatenate(embeddings)))
        pubs_reduced = PCA(n_components=2).fit_transform(pubs_matrix)
    
    print(f'Índice: {index} Vector: ', pubs_reduced[len(pubs_reduced) - 1])
    # Guardar el embedding actual en el archivo
    np.save(output_file, pubs_reduced)