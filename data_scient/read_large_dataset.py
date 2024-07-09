import pandas as pd
from Ejecutor import ejecutar_paralelo

chunksize = 5 * 10**6


dataset_path = '../datasets/visitas.csv'


def task(chunksize):
    print("en el task")
    print(chunksize)
    return 1 * 2


args_list = [chunksize  for _ in range(5)]

print(args_list)

ejecutar_paralelo(num_nucleos=4, task=task, args=args_list)


