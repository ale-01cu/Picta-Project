class EvaluateException(Exception):
    """Excepción lanzada cuando se evalua un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al evaluar el modelo {model_name}.")