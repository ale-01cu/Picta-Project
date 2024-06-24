import matplotlib.pyplot as plt
import pandas as pd


path = "../datasets/vistas_no_nulas.csv"
df = pd.read_csv(path)
pubs_df = pd.read_csv("")


def get_row_by_id(df, id_column, id_value):
    return df.loc[df[id_column] == id_value]

# Cuenta la frecuencia de cada elemento en la columna 'column_x'
counts = df['publicacion_id'].value_counts()


amount = 0
stop = 5

# pubs_df = pubs_df.drop("Unnamed: 0", axis=0)

for id, views in dict(counts).items():
  amount += 1
  print(dict(get_row_by_id(
      df=pubs_df, id_column="id", id_value=id
  )[["id", "nombre"]].values), f"Vistas: {views}")
  if amount == stop: break


# Grafica los resultados
plt.figure(figsize=(8, 6))
counts.plot(kind='bar')
plt.title('Frecuencia de cada elemento en la columna "publicacion_id"')
plt.xlabel('Elemento')
plt.ylabel('Frecuencia')
plt.show()