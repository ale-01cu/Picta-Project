class DataInvalidException(Exception):
    """Excepción lanzada cuando los datos de entrada de la accion use_engine son incorrectors."""
    def __init__(self):
        super().__init__(f"Datos de entrada incorrectos.")