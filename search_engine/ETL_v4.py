import pandas as pd
import spacy
import re
import unicodedata
from spellchecker import SpellChecker
import os
from DBconnect import DBConnect 
from cromaDb import ChromaDBManage 

class NLP:
    def __init__(self):
        # Cargar modelo de Spacy
        self.nlp = spacy.load('es_core_news_sm')
        
        # Inicializar corrector ortográfico
        self.spell = SpellChecker(language='es')

    def is_token_allowed(self, token) -> bool:
        return bool(
            token
            and not token.is_stop  # Stopword Removal
            and token.is_alpha  # Eliminar tokens no alfabéticos
            and not token.is_punct  # Eliminar puntuaciones
        )

    def preprocess_token(self, token) -> str:
        return token.lemma_.strip().lower()  # Lemmatization

    def spell_check(self, text: str) -> str:
        corrected_text = []
        for word in text.split():
            corrected_word = self.spell.correction(word)
            corrected_text.append(corrected_word)
        return ' '.join(corrected_text)

    def filters(self, value):
        if value is None or value == '':
            return ''  # Devuelve una cadena vacía para los valores vacíos
        text = str(value)
        text = re.sub(r'[()\-_]', ' ', text)  # Delimiter Removal
        text = re.sub(r'<[^>]*>', '', text)  # Removal of Tags
        text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
        return text

    def process_text(self, df, columns):
        processed_data = []
        
        for index, row in enumerate(df.iterrows()):
            print(f"Procesando elemento {index}")
            _, row = row  # Desempacar el índice y la fila
            processed_row = {}
            all_empty = True  # Flag para verificar si toda la fila está vacía
            
            for col in columns:
                value = row[col]
                if isinstance(value, str) and value.strip():  # Verifica si el valor es una cadena no vacía
                    tokens = self.nlp(self.filters(value))
                    processed_value = ' '.join([
                        self.preprocess_token(token)
                        for token in tokens
                        if self.is_token_allowed(token)
                    ])
                    all_empty = False  # Si encontramos un valor no vacío, marcamos la fila como no vacía
                elif value is not None:  # Si es None o cualquier otro tipo de dato válido
                    processed_value = str(value)
                    all_empty = False  # Si encontramos un valor no vacío, marcamos la fila como no vacía
                else:  # Si es un valor vacío (None o cadena vacía)
                    processed_value = ''
                
                processed_row[col] = processed_value
            
            # Solo agregamos la fila si no está completamente vacía
            if not all_empty:
                processed_data.append(processed_row)
        
        return pd.DataFrame(processed_data)
    
class ETL:
    def __init__(self, db_connection):
        # Recibir la conexión a la base de datos como un parámetro
        self.db_connection = db_connection
        self.nlp_processor = NLP()  # Instancia de la clase NLP

    def extract(self):
        """Extract data from the database."""
        self.db_connection.connect_mongo()
        raw_data = self.db_connection.extract_data_mongo()
        self.db_connection.close_connection()

        # Convertir los datos crudos en una lista de diccionarios
        documents = []
       
        for doc in raw_data:
            # Asegúrate de que cada documento tenga las claves necesarias
            document = {
                'id': str(doc.get('id', '')),
                'nombre': doc.get('nombre', ''),
                'descripcion': doc.get('descripcion', ''),
                'categoria': doc.get('categoria', '')
            }
           
            documents.append(document)

        return documents

    def transform(self, df):
        """Remove documents with empty fields and apply NLP."""
        # Remove rows with any empty fields
        df = df.dropna(how='any')
        
        # Process the text using NLP for 'nombre' and 'descripcion'
        processed_text_df = self.nlp_processor.process_text(df, ['nombre', 'descripcion'])
        
        # Mantener las columnas 'id' y 'categoria' del DataFrame original
        processed_df = pd.concat([df[['id', 'categoria']].reset_index(drop=True), processed_text_df], axis=1)
        
        return processed_df

    def load_to_chroma(self, df):
        # Crear una instancia de ChromaDBHandler
        chroma_handler = ChromaDBManage()

        # Crear o obtener la colección en Chroma DB
        chroma_handler.create_and_load_documents(df,collection_name="publicaciones")

        # Cargar documentos en Chroma DB
        documents = []
        for i, record in df.iterrows():
            document = {
                'id': str(record['id']),
                'nombre': record['nombre'],
                'descripcion': record['descripcion'],
                'categoria': record['categoria']
            }
            documents.append(document)

        # Usar el método load_documents de ChromaDBHandler
        chroma_handler.load_documents(documents)

    def load_processed_to_mongo(self, df):
        """Save the processed data to a new MongoDB collection."""
        # Cambiar la colección a "Publicaciones Procesadas"
        self.db_connection.collection_name = "Publicaciones Procesadas"
        self.db_connection.connect_mongo()
        self.db_connection.insert_data_mongo(df)
        self.db_connection.close_connection()

    def run(self):
        """Main method to execute the ETL process."""
        # Extract data
        df = self.extract()
        
        # Transform data
        processed_df = self.transform(df)
        
        # Load processed data into MongoDB
        self.load_processed_to_mongo(processed_df)  # Load processed data into the new collection
        
        # Load data into Chroma DB
        self.load_to_chroma(processed_df)  # Load processed data with NLP
