import matplotlib as plt
import numpy as np

embeddings_reduced = np.load('../embeddings/matriz.npy')

print(embeddings_reduced)

# plt.figure(figsize=(100,100))
# plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1])

# # Añadimos anotaciones a cada punto
# for i in range(embeddings_reduced.shape[0]):
#     plt.annotate(f'Pos # {i}', (embeddings_reduced[i, 0], embeddings_reduced[i, 1]))

# plt.show()