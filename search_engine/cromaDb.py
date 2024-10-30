import chromadb
import logging
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import chromadb.config
# Configurar el logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaDBManage:

    default_settings = chromadb.config.Settings(
    is_persistent=True,
    persist_directory="./cromaDB",
    chroma_memory_limit_bytes=1073741824,  # 1 GB
    chroma_server_thread_pool_size=150,    # Aumentar el tamaño del pool de hilos
    allow_reset=True,
    # Otros ajustes según sea necesario
    )
    def __init__(self, collection_name, settings=None):
        # Usar PersistentClient con configuraciones personalizadas o por defecto
        settings = settings or self.default_settings  # Usa configuraciones por defecto si no se proporcionan

        # Asegúrate de que el cliente se inicialice con las configuraciones correctas
        self.client = chromadb.PersistentClient(settings=settings)
        self.collection = self.client.get_or_create_collection(name=collection_name)
       
        logging.info(f"Collection '{collection_name}' created with persistent client.")
        #print(self.default_settings)
    def add_documents(self, documents, ids):
     try:
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            # Verificar si el ID ya existe
            existing_ids = self.collection.get(ids=[doc_id])
            if existing_ids:
                logging.warning(f"Document with ID {doc_id} already exists. Skipping.")
                continue

            logging.debug(f"Adding document {i+1}: ID: {doc_id}")
            self.collection.add(documents=[doc], ids=[doc_id])
        logging.info(f"Added {len(documents)} documents to the collection.")
     except Exception as e:
        logging.error(f"Error adding documents: {e}")

    def count_items(self):
        # Contar los elementos en la colección
        item_count = self.collection.count()
        logging.info(f"Count of items in collection: {item_count}")
        print(f"Count of items in collection: {item_count}")

    def inspect_collection(self):
        # Inspeccionar la colección
        logging.info("Inspecting collection.")
        print(self.collection)

    def delete_all_data(self):
        # Eliminar todos los datos de la colección
        self.client.delete_collection(name="publicacion")

    def showConfig(self):
        settings = chromadb.config.Settings()
        config_dict = vars(settings)  
        for key, value in config_dict.items():
            print(f"{key}: {value}")

    
   
""" client.heartbeat() # returns a nanosecond heartbeat. Useful for making sure the client remains connected.
client.reset() # Empties and completely resets the database. ⚠️ This is destructive and not reversible. """


