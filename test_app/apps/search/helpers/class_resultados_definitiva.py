import math
import nltk
import string
from .clase_resultados_query import Resultados
from operator import itemgetter
from nltk import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer



class Resultados_2_pesos:
  def __init__(self, query, resultados, pesos):
    self.query = query # query
    self.resultados = resultados.copy()
    self.pesos = pesos
    self.resultados_etiquetados_por_id = self.build_resultados_etiquetados_por_id() #Result
    self.doc_terms = {}
    self.qry_terms = {}
    
    # Paso 1
    self.build_terms() #Result

    # # Paso 2
    self.all_terms = self.collect_vocabulary()

    # # Paseo 3
    self.doc_vectors = self.vectorize(self.doc_terms, self.all_terms)
    self.qry_vectors = self.vectorize(self.qry_terms, self.all_terms)

    # # Paso 4
    self.doc_idfs = self.calculate_idfs(self.all_terms, self.doc_terms)


    # # Paso 5
    self.doc_vectors = self.vectorize_idf(self.doc_terms, self.doc_idfs, self.all_terms)
    
    # # Paso 6
    self.queryB = self.qry_vectors.get("1")
    self.results = {}

    for doc_id in self.doc_vectors.keys():
      document = self.doc_vectors.get(doc_id)
      cosine = self.calculate_cosine(self.queryB, document)
      self.results[doc_id] = cosine  

    self.sorted_results = sorted(self.results.items(), key=itemgetter(1), reverse=True)[:44]

  def get_resultados_ordenados(self):
    lista_ordenada = []

    for i in self.sorted_results:
        lista_ordenada.append(self.resultados_etiquetados_por_id[i[0]])

    return lista_ordenada

  # FIXME: quitar el campo id
  def build_resultados_etiquetados_sin_id(self):
    auxiliar = {}
    for res in self.resultados:
      auxiliar[res['id']] = res

    return auxiliar

  def build_resultados_etiquetados_por_id(self):
    auxiliar = {}

    for res in self.resultados:
      auxiliar[res['id']] = res

    return auxiliar


  def build_terms(self):
    dict_resultados = self.resultados_etiquetados_por_id

    for doc_id in dict_resultados.keys():
      terms = {}
      for campo, texto in dict_resultados.get(doc_id).items():
        if campo in self.pesos:
            peso = self.pesos[campo]
            campo_terms = self.get_terms(texto)
            for term, count in campo_terms.items():
              terms[term] = terms.get(term, 0) + count * peso
      self.doc_terms[doc_id] = terms

    self.qry_terms['1'] = self.get_terms(self.query)


  def get_terms(sef, text):
    stoplistS = set(stopwords.words('spanish'))
    stoplistE = set(stopwords.words('english'))
    patron = r'\w+|[^\w\s]'
    terms = {}
    st = SnowballStemmer('spanish')
    word_list = [
        st.stem(word.lower()) for word in regexp_tokenize(text, patron)
        if word.isalnum() 
        and word.lower() not in stoplistS 
        and word.lower() not in stoplistE 
        and word.lower() not in string.punctuation
      ]

    for word in word_list:
      terms[word] = terms.get(word, 0) + 1
    return terms


  def collect_vocabulary(self):
    all_terms = []
    for doc_id in self.doc_terms.keys():
      for term in self.doc_terms.get(doc_id).keys():
        all_terms.append(term)
    for term in self.qry_terms.get("1").keys():
      all_terms.append(term)
    return sorted(set(all_terms))


  def vectorize(self, input_features, vocabulary):
    output = {}
    for item_id in input_features.keys():
      features = input_features.get(item_id)
      output_vector = []
      for word in vocabulary:
        if word in features.keys():
          output_vector.append(int(features.get(word)))
        else:
          output_vector.append(0)
      output[item_id] = output_vector
    return output


  def calculate_idfs(self, vocabulary, doc_features):
    doc_idfs = {}
    doc_counts = {}
    for term in vocabulary:
      doc_counts[term] = 0
      for doc_id in doc_features.keys():
        terms = doc_features.get(doc_id)
        if term in terms.keys():
          doc_counts[term] += terms.get(term)
          if term in self.pesos:
            doc_counts[term] += 1  # Añadir 1 si el término tiene peso
          else:
            doc_counts[term] += 0.01  # Añadir un pequeño valor para evitar divisiones por cero
            doc_idfs[term] = math.log(
              float(len(doc_features.keys())) / float(doc_counts[term]), 10)
    return doc_idfs


  def vectorize_idf(self, input_terms, input_idfs, vocabulary):
    output = {}
    for item_id in input_terms.keys():
      terms = input_terms.get(item_id)
      output_vector = []
      for term in vocabulary:
        if term in terms.keys():
          output_vector.append(
            input_idfs.get(term)*float(terms.get(term)))
        else:
          output_vector.append(float(0))
      output[item_id] = output_vector
    return output



# **********************************************************
  def length(self, vector):
    sq_length = 0
    for index in range(0, len(vector)):
      sq_length += math.pow(vector[index], 2)
    return math.sqrt(sq_length)

  def dot_product(self, vector1, vector2):
    if len(vector1)==len(vector2):
      dot_prod = 0
      for index in range(0, len(vector1)):
        if not vector1[index]==0 and not vector2[index]==0:
          dot_prod += vector1[index]*vector2[index]
      return dot_prod
    else:
      return "Unmatching dimensionality"

  def calculate_cosine(self, query, document):
    cosine = self.dot_product(query, document) / ((self.length(query) * self.length(document)) + 1)
    return cosine
