import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from .RetrievalModel import RetrievalModel
from .data_pipeline import test, train, pubs_ds
from .RankingModel import RankingModel
import os

def use_retrieval_model(user_id):
    try:
        model = RetrievalModel([64])
        loaded = model.load('26')
        return loaded

    except Exception as e:
        model.fit_model()
        ids = model.predict()
        model.save()
        return ids

def use_ranking_model(user_id, ids):
    model = RankingModel([64])
    model.fit_model()
    model.predict(user_id, ids)


if __name__ == "__main__":
    score, ids = use_retrieval_model('26')
    print('Scores', score)
    print('ids', ids)
    # ids = use_retrieval_model()
    # use_ranking_model('26', ids)