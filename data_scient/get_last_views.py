import pandas as pd
import os
dirname = os.path.dirname(__file__)


def get_last_views():
  pubs = pd.read_csv(os.path.join(dirname, "../datasets/picta_publicaciones_crudas.csv"))
  views = pd.read_csv(os.path.join(dirname, "../datasets/vistas_no_nulas.csv"))

  views = views[views['publicacion_id'].isin(pubs['id'])]

  last_million_views = views.tail(1000000)
  last_million_views['usuario_id'] = views['usuario_id'].astype(int)
  last_million_views.to_csv("vistas.csv", index=False)


if __name__ == '__main__':
  get_last_views()