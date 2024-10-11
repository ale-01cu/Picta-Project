from engine.models.RetrievalModel import RetrievalModel
from engine.stages.AbstractStage import AbstractStage

class RetrievalStage(AbstractStage):
    model: RetrievalModel
    name: str

    def __init__(self) -> None:
        self.model = RetrievalModel
        self.name = "retrieval"


    def __str__(self) -> str:
        return f"Stage({self.name})"

    def inputs(self):
        pass

    def outputs(self):
        pass    