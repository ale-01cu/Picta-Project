from pymongo import MongoClient
import pandas as pd

class DBConnect:
    def __init__(self, config, collection, features):
        self.config = config 
        self.features = features 
        self.collection_name = collection  # Nombre de la colección
        
    def connect_mongo(self):
        # Conectar a MongoDB
        self.client = MongoClient(self.config['mongo_uri'])
        self.db = self.client[self.config['db_name']]
        self.collection = self.db[self.collection_name]

    def close_connection(self):
        # Cerrar la conexión a MongoDB
        if self.client:
            self.client.close()

    def check_documents_exist(self, document_ids):
        # Verificar qué documentos ya existen en la colección
        existing_docs = self.collection.find({"id": {"$in": document_ids}})
        existing_ids = {doc["id"] for doc in existing_docs}  # Crear un conjunto de IDs existentes
        return existing_ids

    def insert_data_mongo(self, df):
        # Convertir el DataFrame a una lista de diccionarios
        records = df.to_dict('records')

        # Obtener todos los IDs existentes en la colección
        existing_ids = {doc['id'] for doc in self.collection.find({}, {'id': 1})}  # Solo obtener el campo 'id'

        # Filtrar registros que no existen en la colección
        new_records = [record for record in records if record['id'] not in existing_ids]

        # Insertar solo los nuevos registros
        if new_records:
            try:
                result = self.collection.insert_many(new_records)
                print(f"Documentos insertados: {len(result.inserted_ids)}")
            except Exception as e:
                print(f"Error al insertar en MongoDB: {str(e)}")
        else:
            print("No hay documentos nuevos para insertar.")

    def extract_data_mongo(self):
        """Extraer datos de MongoDB como una lista de diccionarios."""
        try:
            # Asegúrate de incluir el campo 'id' y otros campos necesarios
            fields_to_extract = {col: 1 for col in self.features}
            fields_to_extract['id'] = 1  # Asegúrate de incluir el campo 'id'
        
            # Realizar la consulta a MongoDB
            data = list(self.collection.find({}, fields_to_extract))
        
            # Convertir ObjectId a string si es necesario
            for doc in data:
                if '_id' in doc:
                    del doc['_id']  # Eliminar el campo '_id' si no es necesario

            return data
        except Exception as e:
            print(f"Error al extraer datos de MongoDB: {str(e)}")
            return []  # Retorna una lista vacía en caso de error

    def check_documents_exist(self, document_ids):
        # Realizar una consulta para obtener todos los documentos que ya existen
        existing_docs = self.collection.find({"_id": {"$in": document_ids}})
        existing_ids = {doc["_id"] for doc in existing_docs}  # Crear un conjunto de IDs existentes
        return existing_ids

    def get_processed_publications(db_connection):
        """Extraer documentos de la colección 'Publicaciones Procesadas'."""
        db_connection.connect_mongo()
        documents = db_connection.extract_data_mongo()
        db_connection.close_connection()
        return documents
