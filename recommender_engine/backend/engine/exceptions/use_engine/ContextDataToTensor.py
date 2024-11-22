class ContextDataToTensorException(Exception):
    """Excepción cuando falla la conversion de datos de contexto de entrada a un tensor."""
    def __init__(self):
        super().__init__(f"Error al transformar los datos de contexto en un Tensor.")