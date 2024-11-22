class ConvertToTfDatasetException(Exception):
    """Excepción lanzada cuando no se puede convertir a un dataset de tensorflow"""
    def __init__(self):
        super().__init__(f"Error: Error al convertir a un Dataset de Tensorflow.")