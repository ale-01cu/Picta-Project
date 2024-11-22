class UserIdToTensorException(Exception):
    """Excepción cuando falla la conversion de datos del id del usuario de entrada a un tensor."""
    def __init__(self):
        super().__init__(f"Error al transformar los datos del id del usuario en un Tensor.")