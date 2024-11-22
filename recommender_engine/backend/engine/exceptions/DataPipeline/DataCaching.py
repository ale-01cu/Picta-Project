class DataCachingException(Exception):
    """Excepción personalizada para errores en la función data_caching."""
    def __init__(self, message="Error en la función data_caching"):
        self.message = message
        super().__init__(self.message)