from rest_framework import viewsets, status, generics, filters
from App_Buscador.models import Contenido
from .serializers import SerializerContenido
from rest_framework.response import Response
from App_Buscador.helpers.procesar_texto import search, procesar, aplicar_stop_words
from App_Buscador.helpers.clase_resultados_query import Resultados, diccionario_resultados
from .pagination import ContenidoPagination
import time
from elasticsearch_dsl import function
from App_Buscador.helpers.clase_resultados_query_2 import Resultados_2
from App_Buscador.helpers.class_resultados_definitiva import Resultados_2_pesos
from App_Buscador.searcher.results_controller import ResultsController
from App_Buscador.searcher.query import Query

# def promedioDocument():
#      # Obtener la longitud promedio de los documentos en cada campo
#     avg_titulo_len = function.AvgField(field='nombre.length')
#     avg_descripcion_len = function.AvgField(field='descripcion.length')
#     avg_generos_len = function.AvgField(field='canal.nombre.length')
#     avg_categoria_len = function.AvgField(field='palabraClave.length')

#     # Crear las funciones de puntuación usando la longitud promedio de los documentos en cada campo
#     funciones = [
#         function.FieldValueFactor(field='nombre.length', modifier='none', factor=1.0/avg_titulo_len),
#         function.FieldValueFactor(field='descripcion.length', modifier='none', factor=1.0/avg_descripcion_len),
#         function.FieldValueFactor(field='canal.nombre.length', modifier='none', factor=1.0/avg_generos_len),
#         function.FieldValueFactor(field='palabraClave.length', modifier='none', factor=1.0/avg_categoria_len),
#     ]
#     return funciones  

# def query_exacta(query,):
    
#     funciones=promedioDocument()
#     # Crear la consulta de coincidencias exactas con la cláusula function_score
#     q_exacta = Q('function_score',
#         query=Q('multi_match', query=query, fields=['nombre^4', 'descripcion^1', 'palabraClave^3', 'canal.nombre^2']),
#         functions=funciones,
#         boost_mode='replace'
#     )

#     return q_exacta

# def query_degradada(query,):
    
#     funciones=promedioDocument()
#     # Crear la consulta de coincidencias parciales con la cláusula function_score
#     q_degradada = Q('function_score',
#         query=Q('more_like_this', like=query, fields=['nombre^4', 'canal.nombre^2', 'palabraClave^3', 'descripcion^0.5']),
#         functions=funciones,
#         boost_mode='replace'
#     )

#     return q_degradada

# def query_wildcard(query):
#     # Obtener las funciones de puntuación
#     funciones = promedioDocument()

#     # Crear la consulta de coincidencias parciales con la cláusula function_score
#     q_wildcard = Q('function_score', boost_mode='replace').query(
#         'multi_match', query=query, fields=['nombre^4', 'descripcion^1', 'canal.nombre^2', 'palabraClave^3']
#     )
#     q_wildcard = q_wildcard.functions(funciones)

#     return q_wildcard


class ContenidoView(viewsets.ModelViewSet):
    queryset = Contenido.objects.all()
    serializer_class = SerializerContenido
    pagination_class = ContenidoPagination

from elasticsearch_dsl import Search, SF
from elasticsearch_dsl import Q

class BusquedaView(viewsets.ViewSet):
   
  def list(self, request, busqueda):
      start_time = time.time()

      query_busqueda = request.query_params.get('busqueda', None).lower()
      procesada = ' '.join(procesar(query_busqueda))
      limpias = aplicar_stop_words(query_busqueda)

      q = Q(
          'bool',
          should=[
              Q('match_phrase', titulo={'query': limpias, 'slop': 1, 'boost': 6}),
              Q('multi_match', query=limpias, fields=[
                'titulo^4', 'categoria^3', 'generos^2', 'descripcion^0.2'
                ], boost=5, fuzziness='AUTO'),
              Q('more_like_this', like=procesada, fields=[
                'titulo^4', 'categoria^3', 'generos^2', 'descripcion^0.2'
                ], boost=4),
          ],
          minimum_should_match=1,
          )

      # Agregar la búsqueda wildcard para tener en cuenta las palabras que contienen la consulta
      for palabra in procesada.split():
          q |= Q('wildcard', nombre={'value': f'*{palabra}*'}) | Q('wildcard', categoria={'value': f'*{palabra}*'}) | Q('wildcard', generos={'value': f'*{palabra}*'}) | Q('wildcard', descripcio={'value': f'*{palabra}*'})

      s = Search(index='contenido').query(q)
      response = s.execute()

      resultados = []
      for hit in response.hits:
          resultados.append(hit.to_dict())

      pesos = {
        "titulo": 0.4,
        "descripcion": 0.1,
        "generos": 0.3,
        "categoria": 0.2,
      }


      query_object:Query = Query(query_busqueda)
      query_procesed:str = query_object.query_terms_joined
      results = ResultsController(
        query_object, 
        response.hits, 
        pesos
      )
      docs_terms:dict = results.build_docs_terms()
      query_terms:dict = query_object.query_terms
      vocabulary:dict = results.collect_vocabulary(docs_terms)
      docs_vertors:dict = results.vectorize(docs_terms, vocabulary)
      query_vectors:dict = results.vectorize(query_terms, vocabulary).get("1")
      docs_idfs_vectors:dict = results.calculate_idf_verctors(docs_terms, vocabulary)

      cosine_similarity_docs:dict = results.calculate_cosine_similarity(
        query_vectors, 
        docs_idfs_vectors
      )
      
      sorted_results = results.sorted_results(cosine_similarity_docs)
      response = results.sorted_docs(sorted_results)


      # res = Resultados_2(procesada, resultados)
      # res2 = Resultados_2_pesos(procesada, resultados, pesos)
      # response = res2.get_resultados_ordenados()

      end_time = time.time()
      tiempo_total = round(end_time - start_time, 2)
      print(f"La busqueda tardo {tiempo_total} en ejecutarse.")

      return Response(
        response, 
        status=status.HTTP_200_OK
      )


  # def list(self, request, busqueda):
  #     query_busqueda = request.query_params.get('busqueda', None).lower()
  #     procesada = ' '.join(procesar(query_busqueda))
  #     limpias = aplicar_stop_words(query_busqueda)

  #     q = Q(
  #         'bool',
  #         should=[
  #             Q('match_phrase', nombre={'query': procesada, 'slop': 1}),
  #             Q('multi_match', query=procesada, fields=['nombre^4', 'canal.nombre^2', 'palabraClave¨3', 'descripcion^1']),
  #             Q('more_like_this', like=procesada, fields=['nombre^4', 'canal.nombre^2', 'palabraClave¨3', 'descripcion^1']),
  #         ],
  #         minimum_should_match=1,
  #         )

  #     # Agregar la búsqueda wildcard para tener en cuenta las palabras que contienen la consulta
  #     for palabra in procesada.split():
  #         q |= Q('wildcard', nombre={'value': f'*{palabra}*'}) #| Q('wildcard', categoria={'value': f'*{palabra}*'}) | Q('wildcard', generos={'value': f'*{palabra}*'}) | Q('wildcard', descripcio={'value': f'*{palabra}*'})

  #     s = Search(index='publicacion-index').query(q)
  #     response = s.execute()

  #     resultados = []
  #     for hit in response.hits:
  #         resultados.append(hit.to_dict())

  #     return Response(resultados, status=status.HTTP_200_OK)



    # resultados_por_querys = {}
  
    # # Se obtiene el tiempo en el que se inicia la ejecución de la búsqueda
    # start_time = time.time()

    # # Se obtiene el parámetro "busqueda" de la URL y se convierte a minúsculas
    # query_busqueda = request.query_params.get('busqueda', None).lower()
    # query_procesada = procesar(query_busqueda)
    # query_procesada_texto = " ".join(query_procesada)
    
    # if " ".join(query_procesada) in diccionario_resultados.keys():
    #   return Response(diccionario_resultados[query_procesada_texto].resultados, status=status.HTTP_200_OK)

    # querySets = Contenido.objects.all()
    # serializer = SerializerContenido(querySets, many=True)

    # resultado = []
          
    # # Se itera sobre los objetos serializados para buscar coincidencias con la búsqueda
    # for i in serializer.data:
    #   todo = f"{i['categoria']}s {i['titulo']} {i['descripcion']} {i['fecha_de_estreno']} {i['generos']}".lower()
    #   coincidencias = search(query_busqueda, todo)
    #   if coincidencias > 0: resultado.append({"e": i, "c":coincidencias})
      
    # # Se ordenan los resultados por número de coincidencias, de mayor a menor
    # ordenados = sorted(resultado, key=lambda x: x['c'], reverse=True)
    # # Se crea una lista solo con los objetos ordenados (sin el número de coincidencias)
    # respuesta = [e['e'] for e in ordenados]
    
    # nuevo_resultado = Resultados(query_procesada, respuesta)
    # nuevo_resultado.entrenar()
    # respuesta = nuevo_resultado.resultados.copy()
    
    # diccionario_resultados[" ".join(query_procesada)] = nuevo_resultado
  
    # # Se obtiene el tiempo en el que se termina la ejecución de la búsqueda
    # end_time = time.time()
    # # Se calcula el tiempo total de ejecución
    # tiempo_total = round(end_time - start_time, 2)
    # respuesta.append(tiempo_total)
    # # Se imprime el tiempo total de ejecución
    # print(f"La busqueda tardo {tiempo_total} segundos en ejecutarse")
    
    # # Se devuelve la lista de objetos ordenados como respuesta
    # return Response(respuesta, status=status.HTTP_200_OK)