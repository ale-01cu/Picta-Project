import matplotlib.pyplot as plt
import numpy as np

embeddings_reduced = np.load('../datasets/pubs_embeddings.npy')

print(len(embeddings_reduced))


def show_2d():
    plt.figure(figsize=(100,100))
    plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1])

    # Añadimos anotaciones a cada punto
    for i in range(embeddings_reduced.shape[0]):
        plt.annotate(f'Pos # {i}', (embeddings_reduced[i, 0], embeddings_reduced[i, 1]))

    plt.show()


def show_3d():
    fig = plt.figure(figsize=(100,100))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], embeddings_reduced[:, 2])

    # Añadimos anotaciones a cada punto
    for i in range(embeddings_reduced.shape[0]):
        ax.text(embeddings_reduced[i, 0], embeddings_reduced[i, 1], embeddings_reduced[i, 2], f'Pos # {i}')

    plt.show()

show_3d()