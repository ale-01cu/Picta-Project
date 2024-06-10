from sentence_transformers import SentenceTransformer, util


models = ['all-MiniLM-L6-v2', 'sentence-transformers/stsb-roberta-large', 'all-mpnet-base-v2']

print('Descargando el modelo...')
model = SentenceTransformer(models[0])
print('El modelo se descargo correctamente.')