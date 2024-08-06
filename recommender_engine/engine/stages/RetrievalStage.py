from recommender_engine.engine.models.RetrievalModel import RetrievalModel
from stages.Stages import Stages

class RetrievalStage(Stages):
    retrieval_model: RetrievalModel
    _instance = None

    def __init__(self) -> None:
        self.retrieval_model = RetrievalModel

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RetrievalStage, cls).__new__(cls)
            # cls._instance.retrieval_model = RetrievalModel
        return cls._instance

    def inputs(self):
        pass

    def outputs(self):
        pass


RetrievalStage()