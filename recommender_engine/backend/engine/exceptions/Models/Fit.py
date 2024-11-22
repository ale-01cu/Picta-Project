class FitException(Exception):
    """Excepción lanzada cuando se entrena un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al entrenar el modelo {model_name}.")