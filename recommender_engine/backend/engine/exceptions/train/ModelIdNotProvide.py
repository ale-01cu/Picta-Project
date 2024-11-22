class ModelIdNotProvideException(Exception):
    """Excepción lanzada cuando no se encuentra un modelo."""
    def __init__(self, model_name):
        super().__init__(f"No se provee el id para el modelo {model_name}.")