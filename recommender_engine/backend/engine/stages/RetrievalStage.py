from engine.models.RetrievalModel import RetrievalModel
from engine.stages.AbstractStage import AbstractStage

class RetrievalStage(AbstractStage):
    _instance = None  # Atributo de clase para almacenar la instancia única
    model: RetrievalModel
    name: str

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RetrievalStage, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, 'initialized'):  # Verifica si ya ha sido inicializado
            self.model = RetrievalModel
            self.name = "retrieval"
            self.initialized = True  # Marca como inicializado

    def __str__(self) -> str:
        return f"Stage({self.name})"

    def inputs(self):
        pass

    def outputs(self):
        pass 