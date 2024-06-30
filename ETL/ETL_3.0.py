import pandas as pd
from pandas import DataFrame
import spacy
import re
import unicodedata

nlp = spacy.load('es_core_news_lg')
pubs_path = '/content/publicaciones_crudas.csv'
pubs_output_path = '/content/publicaciones_procesadas_sin_nulas.csv'
features = ['nombre', 'descripcion', 'categoria']

def is_token_allowed(token: str) -> bool:
  return bool(
      token
      and not token.is_stop
      and token.is_alpha
      and not token.is_punct
  )

def preprocess_token(token: str) -> bool:
  return token.lemma_.strip().lower()


def filters(value):
  text = str(value)
  text = re.sub(r'[()\-_]', ' ', text)
  text = ''.join((c for c in unicodedata.normalize(
     'NFD', value) if unicodedata.category(c) != 'Mn'))
  return text

def process_text(df, columns):
    df[columns] = df[columns].applymap(
      lambda value: str(' '.join([
        preprocess_token(token)
        for token in nlp(filters(value))
        if is_token_allowed(token)
    ])))
    return df


print("Cargando las Publicaciones...")
df = pd.read_csv(pubs_path)
print("Forma: ", df.shape)

print("Filas nulas: ")
print(df[df.isnull().any(axis=1)])


print("Eliminando Filas nulas...")
before = df.shape[0]
df = df.dropna()
after = df.shape[0]
delete_amount = before - after

print("Filas antes: ", before) 
print("Filas despues: ", after)
print("Se eliminaron: ", delete_amount, "Filas.")


print("Procesando el texto...")
new_df = process_text(df.copy(), features)
new_df.to_csv(pubs_output_path, index=False)


df = pd.read_csv(pubs_output_path)
df = df.dropna()
df.to_csv(pubs_output_path, index=False)



