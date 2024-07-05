import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter
import numpy as np

def aplicar_stop_words(texto):
  stop_words = set(stopwords.words('spanish'))
  patron = r'\w+|[^\w\s]'
  tokens = regexp_tokenize(texto, patron)
  tokens_limpios = [ token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words ] 
  return " ".join(tokens_limpios)

def procesar(texto):
  stop_words = set(stopwords.words('spanish'))

  # Tokenizar la frase
  #tokens = word_tokenize(texto)
  patron = r'\w+|[^\w\s]'
  tokens = regexp_tokenize(texto, patron)
  stemmer = SnowballStemmer('spanish')

  # Eliminar las palabras vacías
  tokens_limpios = [ stemmer.stem(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words ] 
  return set(tokens_limpios)


def cosine_similarity(query, document):
    query_tokens = procesar(query)
    document_tokens = procesar(document)
    
    # Calcular la frecuencia de las palabras en la consulta y el documento
    query_freq = dict(nltk.FreqDist(query_tokens))
    document_freq = dict(nltk.FreqDist(document_tokens))
        
    # Obtener un conjunto único de todas las palabras en la consulta y el documento
    all_words = set(query_freq.keys()).union(set(document_freq.keys()))
    
    # Crear vectores de frecuencia para la consulta y el documento
    query_vector = np.array([query_freq.get(word, 0) for word in all_words])
    document_vector = np.array([document_freq.get(word, 0) for word in all_words])
    
    # Calcular la similitud coseno entre los vectores de frecuencia
    dot_product = np.dot(query_vector, document_vector)
    query_norm = np.linalg.norm(query_vector)
    document_norm = np.linalg.norm(document_vector)
    similarity = dot_product / (query_norm * document_norm)
    
    return similarity


def search(query, document):
  document_procesado = procesar(document)
  
  coincidencias = 0

  print("query procesada ***", query)
  print("documento procesado ***", document_procesado)
  similitud = cosine_similarity(query, document)
  
  
  mensaje = f'Comparacion:\n query: {query} \n doc: {document_procesado} \n coincidencias {similitud}'
  
  return similitud
