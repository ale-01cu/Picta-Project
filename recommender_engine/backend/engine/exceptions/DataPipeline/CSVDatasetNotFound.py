class CSVDatasetNotFoundException(Exception):
    """Excepción lanzada cuando no se encuentra un fichero .csv."""
    def __init__(self, path):
        super().__init__(f"Error: El archivo {path} no se encontró.")