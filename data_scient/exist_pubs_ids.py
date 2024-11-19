import pandas as pd
import os
dirname = os.path.dirname(__file__)

def exist_pubs_ids():
  pubs = pd.read_csv(os.path.join(dirname, "../datasets/picta_publicaciones_crudas.csv"))
  views = pd.read_csv(os.path.join(dirname, "../datasets/vistas_no_nulas.csv"))

  ids_no_existentes = views[~views['publicacion_id'].isin(pubs['id'])]

  # Mostrar los IDs que no existen
  if not ids_no_existentes.empty:
      print("IDs de publicaciones que no existen en el dataset de publicaciones:")
      print(ids_no_existentes['publicacion_id'].unique())
      print(f"Cantidad de IDs que no existen: {len(ids_no_existentes['publicacion_id'].unique())}")
  else:
      print("Todos los IDs de vistas existen en el dataset de publicaciones.")


if __name__ == "__main__":
  exist_pubs_ids()