class GetLengthsException(Exception):
    def __init__(self, message="Error al obtener las longitudes del dataset"):
        self.message = message
        super().__init__(self.message)