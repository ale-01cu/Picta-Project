class IndexingException(Exception):
    """Excepción lanzada cuando se indexa un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al indexar el modelo {model_name}.")