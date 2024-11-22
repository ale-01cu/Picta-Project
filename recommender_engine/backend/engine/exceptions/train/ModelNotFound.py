class ModelNotFoundException(Exception):
    """Excepción lanzada cuando no se encuentra un modelo."""
    def __init__(self, model_id):
        super().__init__(f"Modelo con ID {model_id} no encontrado.")