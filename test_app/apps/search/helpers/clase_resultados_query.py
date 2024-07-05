import math
from App_Buscador.models import Contenido
from App_Buscador.api.serializers import SerializerContenido
from App_Buscador.helpers.procesar_texto import procesar

def get_todos_documentos_etiquetados():
  # Se obtienen todos los objetos Contenido de la base de datos
  querySets = Contenido.objects.all()
  # Se serializan los objetos obtenidos para poder manipularlos en Python
  serializer = SerializerContenido(querySets, many=True)
  
  resultadosEtiquetados = {}
  
  for i in serializer.data:
    resultadosEtiquetados[i['id']] = f"{i['categoria']}s {i['titulo']} {i['descripcion']} {i['fecha_de_estreno']} {i['generos']}".lower()
    
  return resultadosEtiquetados
    


class Resultados():
  def __init__(self, query, resultados):
    # La query se recive ya procesada desde fuera
    self.query = query
    self.resultados = resultados.copy()
    self.resultadosEtiquetados = {}
    self.interacciones = {}
    self.tf_idf_vectores_resultados = {}
    self.suma_vectores = {}
    self.documentos_etiquetados = get_todos_documentos_etiquetados()
    self.promedio_suma_vectores = 0
  

  def get_resultados(self):
    return self.resultados


  def armar_interacciones(self):
    for res in self.resultados:
      self.interacciones[str(res['id'])] = 0
  

  def etiquetarResultados(self):
    for i in self.resultados:
      self.resultadosEtiquetados[i['id']] = f"{i['categoria']}s {i['titulo']} {i['descripcion']} {i['generos']}".lower()
    

  def sumar_vectores(self):
    for id, vector in self.tf_idf_vectores_resultados.items():
      if str(id) in self.interacciones.keys():
        self.suma_vectores[id] = sum(vector) + self.interacciones[str(id)]
      
    self.ordenar_vectores()


  def ordenar_vectores(self):
    self.suma_vectores = dict(sorted(self.suma_vectores.items(), key=lambda x: x[1], reverse=True))

    
  def actualizar_vectores(self, id:int):
    print(type(id), id)
    print("vectores***********", self.suma_vectores)
    self.suma_vectores[id] += self.promedio_suma_vectores
    self.ordenar_vectores()
    print(self.suma_vectores)

  def actualizar_resultados(self):
    resultados_auxiliar = []

    
    for i, e in enumerate(self.suma_vectores):
      id_elemento = e
      
      # Busca el id en los resultados
      for pos, res in enumerate(self.resultados):
        if res['id'] == id_elemento:
          resultados_auxiliar.append(res)
          break
      
    for i, ob in enumerate(self.resultados):
      self.resultados[i] = resultados_auxiliar[i]
      
  def filtrar_documentos(self):
    aux = dict(filter(lambda x: str(x[0]) in self.interacciones.keys(), self.tf_idf_vectores_resultados.items()))
    self.tf_idf_vectores_resultados = aux
  

  def calcular_promedio_suma_vectores(self):
    tamaño = len(self.suma_vectores)
    suma_total = 0
    
    for key, suma in self.suma_vectores.items():
      suma_total += suma

    if suma_total == 0 and tamaño == 0:
      self.promedio_suma_vectores = 0
    else:
      self.promedio_suma_vectores = (suma_total / tamaño)
            
    
  def entrenar(self):
    if len(self.resultados) > 0:
      self.armar_interacciones()
      self.etiquetarResultados()
      self.calculate_tf_idf(self.documentos_etiquetados)
      self.filtrar_documentos()
      self.sumar_vectores()
      self.actualizar_resultados()
      self.calcular_promedio_suma_vectores()
      
      
     
  def calculate_idfs(self, vocabulary, doc_features):
    # crea un diccionario vacío para almacenar los IDFs
    doc_idfs = {}
    # itera por cada término del vocabulario
    for term in vocabulary:
      doc_count = 0
      # itera por cada documento en el conjunto de documentos
      for doc_id in doc_features.keys():
        # obtiene los términos del documento actual
        terms = doc_features.get(doc_id)
        # si el término actual aparece en el documento actual, aumenta el contador
        if term in terms.keys():
          doc_count += 1
      # calcula el IDF del término actual y lo almacena en el diccionario de IDFs
      doc_idfs[term] = math.log(
        float(len(doc_features.keys()))/
        float(1 + doc_count), 10)
    # devuelve el diccionario de IDFs
    return doc_idfs
  
  def calculate_tf_idf(self, corpus):
    # crea un diccionario para almacenar los términos y su frecuencia en cada documento
    doc_features = {}
    # crea una lista para almacenar todos los términos únicos en todos los documentos
    vocabulary = []
    # itera por cada documento en el corpus
    for doc_id, doc in corpus.items():
      # crea un diccionario vacío para almacenar los términos y su frecuencia en el documento actual
      doc_procesado = procesar(doc)
      term_freq = {}
      # itera por cada palabra en el documento actual
      for word in doc_procesado:
        if word in self.query:
          # si la palabra actual ya está en el diccionario de términos y frecuencias, aumenta su frecuencia en 1
          if word in term_freq.keys():
            term_freq[word] += 1
          # si la palabra actual no está en el diccionario de términos y frecuencias, agrega una nueva entrada con una frecuencia de 1
          else:
            term_freq[word] = 1
            # si la palabra actual no está en la lista de vocabulario, agréguela
            if word not in vocabulary:
              vocabulary.append(word)
      # agrega el diccionario de términos y frecuencias del documento actual al diccionario de características de documentos
      doc_features[doc_id] = term_freq
    
    # calcula los valores IDF para cada término en el vocabulario
    idfs = self.calculate_idfs(vocabulary, doc_features)
    
    # crea un diccionario para almacenar los vectores de características de cada documento
    doc_vectors = {}
    # itera por cada documento en el corpus
    for doc_id, doc in corpus.items():
      # crea un vector de características vacío para el documento actual
      doc_vector = []
      # itera por cada término en el vocabulario
      for term in vocabulary:
        # si el término actual está presente en el documento actual, calcula su puntaje TF-IDF y lo agrega al vector de características
        if term in doc_features[doc_id].keys():
          tf_idf = doc_features[doc_id][term] * idfs[term]
          doc_vector.append(tf_idf)
        # si el término actual no está presente en el documento actual, agrega un 0 al vector de características
        else:
          doc_vector.append(0)
      # agrega el vector de características del documento actual al diccionario de vectores de características
      doc_vectors[doc_id] = doc_vector
    
    # devuelve el diccionario de vectores de características
    self.tf_idf_vectores_resultados = doc_vectors

  
  def actualizarInteraccion(self, id_resultado):  
    self.interacciones[id_resultado] += 1
    self.actualizar_vectores(int(id_resultado))
    self.actualizar_resultados()
          
            
  # def calculate_idfs(self, vocabulary, doc_features):
  #   print("voabulario ********", vocabulary)
  #   print("caracteristicas ********", doc_features)
  #   # crea un diccionario vacío para almacenar los IDFs
  #   doc_idfs = {}
  #   # itera por cada término del vocabulario
  #   for term in vocabulary:
  #     doc_count = 0
  #     # itera por cada documento en el conjunto de documentos
  #     for doc_id in doc_features.keys():
  #       # obtiene los términos del documento actual
  #       terms = doc_features.get(doc_id)
  #       # si el término actual aparece en el documento actual, aumenta el contador
  #       if term in terms.keys():
  #         doc_count += 1
  #     # calcula el IDF del término actual y lo almacena en el diccionario de IDFs
  #     doc_idfs[term] = math.log(
  #       float(len(doc_features.keys()))/
  #       float(1 + doc_count), 10)
  #   # devuelve el diccionario de IDFs
  #   return doc_idfs

  # def calculate_tf_idf(self, corpus):
  #   # crea un diccionario para almacenar los términos y su frecuencia en cada documento
  #   doc_features = {}
  #   # crea una lista para almacenar todos los términos únicos en todos los documentos
  #   vocabulary = []
  #   # itera por cada documento en el corpus
  #   for doc_id, doc in corpus.items():
  #     # crea un diccionario vacío para almacenar los términos y su frecuencia en el documento actual
  #     term_freq = {}
  #     # itera por cada palabra en el documento actual
  #     for word in doc.split():
  #       # si la palabra actual ya está en el diccionario de términos y frecuencias, aumenta su frecuencia en 1
  #       if word in term_freq.keys():
  #         term_freq[word] += 1
  #       # si la palabra actual no está en el diccionario de términos y frecuencias, agrega una nueva entrada con una frecuencia de 1
  #       else:
  #         term_freq[word] = 1
  #         # si la palabra actual no está en la lista de vocabulario, agréguela
  #         if word not in vocabulary:
  #           vocabulary.append(word)
  #     # agrega el diccionario de términos y frecuencias del documento actual al diccionario de características de documentos
  #     doc_features[doc_id] = term_freq
    
  #   # calcula los valores IDF para cada término en el vocabulario
  #   idfs = self.calculate_idfs(vocabulary, doc_features)
    
  #   # crea un diccionario para almacenar los vectores de características de cada documento
  #   doc_vectors = {}
  #   # itera por cada documento en el corpus
  #   for doc_id, doc in corpus.items():
  #     # crea un vector de características vacío para el documento actual
  #     doc_vector = []
  #     # itera por cada término en el vocabulario
  #     for term in vocabulary:
  #       # si el término actual está presente en el documento actual, calcula su puntaje TF-IDF y lo agrega al vector de características
  #       if term in doc_features[doc_id].keys():
  #         tf_idf = doc_features[doc_id][term] * idfs[term]
  #         doc_vector.append(tf_idf)
  #       # si el término actual no está presente en el documento actual, agrega un 0 al vector de características
  #       else:
  #         doc_vector.append(0)
  #     # agrega el vector de características del documento actual al diccionario de vectores de características
  #     doc_vectors[doc_id] = doc_vector
    
  #   # devuelve el diccionario de vectores de características
  #   self.tf_idf_vectores_resultados = doc_vectors
  



diccionario_resultados = {}

  
    