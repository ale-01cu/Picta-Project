class SaveException(Exception):
    """Excepción lanzada cuando se salva un modelo."""
    def __init__(self, model_name):
        super().__init__(f"Error: Error al salvar el modelo {model_name}.")