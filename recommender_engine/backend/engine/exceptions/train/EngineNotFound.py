class EngineNotFoundException(Exception):
    """Excepción lanzada cuando no se encuentra un motor."""
    def __init__(self, engine_id):
        super().__init__(f"Engine con ID {engine_id} no encontrado.")