class LoadDatasetException(Exception):
    """Excepción personalizada para errores en la función load_dataset."""
    def __init__(self, message="Error en la función load_dataset"):
        self.message = message
        super().__init__(self.message)