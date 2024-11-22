class LoadException(Exception):
    """Excepción lanzada cuando se carga un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al cargar el modelo {model_name}.")