class UpdateDBException(Exception):
    """Excepción lanzada cuando se trata de actualizar la base de datos."""
    def __init__(self, model_name):
        super().__init__(f"Error al tratar de actualizar la coleccion del modelo: {model_name}")