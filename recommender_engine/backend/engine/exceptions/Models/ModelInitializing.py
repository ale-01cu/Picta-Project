class ModelInitializingException(Exception):
    """Excepción lanzada cuando se inicializa un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al inicializar el modelo {model_name}.")