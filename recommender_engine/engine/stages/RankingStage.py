from models.LikesModel import LikesModel
from stages.AbstractStage import AbstractStage

class RankingStage(AbstractStage):
    likes_model: LikesModel

    def __init__(self) -> None:
        self.likes_model = LikesModel
    
    def inputs(self):
        pass

    def outputs(self):
        pass