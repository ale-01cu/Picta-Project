class FeatureNotFoundException(Exception):
    """Excepción lanzada cuando fantan datos de entrada."""
    def __init__(self, key):
        super().__init__(f"Falta el dato de contexto {key}.")