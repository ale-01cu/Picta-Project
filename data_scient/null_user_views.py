import pandas as pd
import time

chunksize = 10 ** 6

start = time.time()


for chunk in pd.read_csv('I:/UCI/tesis/Picta-Project/datasets/visitas.csv', chunksize=chunksize):
    print(chunk)

# print(df.head())
# print('----------')
# print(df.info())
# print('----------')
# print(df.describe())

end = time.time()

print(round(end - start, 2))