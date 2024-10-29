from engine.stages.RetrievalStage import RetrievalStage
from engine.stages.RankingStage import RankingStage
from engine.stages.ReRankingStage import ReRankingStage

retrieval_stage = RetrievalStage()
ranking_Stage = RankingStage()
re_ranking_stage = ReRankingStage()

stages = [
    retrieval_stage.name,
    ranking_Stage.name,
    re_ranking_stage.name,
]
