from ETL_v4 import ETL
from DBconnect import DBConnect
from cromaDb import ChromaDBManage
import logging
import pandas as pd
import chromadb.config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def interactive_search(chroma_handler, db_connection):
    while True:
        query_text = input("Ingrese su consulta de búsqueda (o 'salir' para terminar): ")
        if query_text.lower() == 'salir':
            break
        
        # Realizar búsqueda en Chroma DB
        results = chroma_handler.search_by_text(query_text)
        
        # Imprimir los resultados completos para depuración
        print(results)
        
        # Verificar si hay resultados
        if results['ids'] and results['ids'][0]:
            result_ids = results['ids'][0]
            
            # Conectar a MongoDB y buscar los documentos completos
            db_connection.connect_mongo()
            documents = db_connection.collection.find({"id": {"$in": result_ids}})
           
            
            # Mostrar los documentos completos
            for doc in documents:
                print(f"ID: {doc['id']}, Nombre: {doc['nombre']}, Descripción: {doc['descripcion']}, Categoría: {doc['categoria']}")

            db_connection.close_connection()

def main():
    # Configuración para la conexión a MongoDB
    """  db_config = {
            'mongo_uri': 'mongodb://localhost:27017',
            'db_name': 'Picta'
        }
        db_connection = DBConnect(db_config, 'Publicaciones Procesadas', ['id', 'nombre', 'descripcion', 'categoria'])

        # Ejecutar el proceso ETL
        etl_processor = ETL(db_connection)
        df = etl_processor.extract() """
    #print(df.head(10))

    logging.info("Starting main process.")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    """ df = pd.read_csv('./datasets/picta_publicaciones_procesadas_sin_nulas.csv')

    # Suponiendo que el CSV tiene columnas 'document' e 'id'
    documents = df['nombre'].tolist()
    ids = df['id'].astype(str).tolist()  """

    # Crear una instancia de ChromaDBManage
    chroma_handler = ChromaDBManage("publicaciones")
    #chroma_handler.showConfig()
    #chroma_handler.client.delete_collection("publicaciones")
    chroma_handler.count_items()

    all_ids = chroma_handler.collection.get()
    print(f"Todos los IDs en la colección: {all_ids['ids']}")
    #chroma_handler.client.reset()
    
    # Agregar documentos a la colección
    """ for i in range(len(documents)):
        print(f"Documento {i+1}: ID: {ids[i+1]}, Nombre: {documents[i+1]}")
        chroma_handler.add_documents([documents[i]], [ids[i]]) """    

    logging.info("Main process completed.")

if __name__ == "__main__":
    main()