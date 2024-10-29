import pandas as pd
import spacy
import re
import unicodedata
from spellchecker import SpellChecker
import os
from search_engine.DBconnect import DBConnect 
from search_engine import ChromaDBHandler 

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
    def __init__(self):
        # Configuración para la conexión a MongoDB
        self.db_config = {
            'mongo_uri': 'mongodb://localhost:27017',
            'db_name': 'Picta'
        }
        self.db_connect = DBConnect(self.db_config, 'Publicaciones', ['id', 'nombre', 'descripcion', 'categoria'])
        self.nlp_processor = NLP()  # Instancia de la clase NLP
        self.chroma_collection = None  # Inicializar la colección de Chroma DB

    def extract(self, csv_path):
        """Read the CSV file and return a DataFrame."""
        df = pd.read_csv(csv_path)
        return df

    def transform(self, df):
        """Remove documents with empty fields and apply NLP."""
        # Remove rows with any empty fields
        df = df.dropna(how='any')
        
        # Process the text using NLP
        processed_df = self.nlp_processor.process_text(df, ['nombre', 'descripcion'])
        return processed_df

    def load_to_mongo(self, df):
        """Save the data to MongoDB without applying NLP."""
        self.db_connect.insert_data_mongo(df)

    def load_to_chroma(self, df):
    # Crear una instancia de ChromaDBHandler
        chroma_handler = ChromaDBHandler()  # Asegúrate de que la clase ChromaDBHandler esté importada

        # Crear o obtener la colección en Chroma DB
        chroma_handler.create_chroma_collection(collection_name="publicaciones")

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

    def run(self, csv_path):
        """Main method to execute the ETL process."""
        # Extract data
        df = self.extract(csv_path)
        
        # Transform data
        processed_df = self.transform(df)
        
        # Load data into MongoDB
        self.load_to_mongo(df)  # Load original data without NLP
        # Load data into Chroma DB
        self.load_to_chroma(processed_df)  # Load processed data with NLP

if __name__ == "__main__":
    etl_processor = ETL()
    etl_processor.run('./datasets/picta_publicaciones_crudas.csv')  


