from engine.stages.AbstractStage import AbstractStage

class ReRankingStage(AbstractStage):
    model: object
    name: str

    def __init__(self) -> None:
        self.model = None
        self.name = "re_ranking"

    def __str__(self) -> str:
        return f"Stage({self.name})"

    def inputs(self):
        pass

    def outputs(self):
        pass