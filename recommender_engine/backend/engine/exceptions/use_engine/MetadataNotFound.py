class MetadataNotFoundException(Exception):
    """Excepción lanzada cuando no se encuentran los metadatos."""
    def __init__(self, model_name):
        super().__init__(f"Metadata del Modelo con el nombre {model_name} no encontrada.")