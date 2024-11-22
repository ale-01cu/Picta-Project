class ModelInferenceException(Exception):
    """Excepción cuando falla la inferencia de un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error al realizar la inferencia del modelo {model_name}.")