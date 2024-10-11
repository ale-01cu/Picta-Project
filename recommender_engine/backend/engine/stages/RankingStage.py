from engine.models.LikesModel import LikesModel
from engine.stages.AbstractStage import AbstractStage

class RankingStage(AbstractStage, ):
    model: LikesModel
    name: str

    def __init__(self) -> None:
        self.model = LikesModel
        self.name = "ranking"

    def __str__(self) -> str:
        return f"Stage({self.name})"

    def inputs(self):
        pass

    def outputs(self):
        pass