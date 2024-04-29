from .RetrievalModel import RetrievalModel
from .RankingModel import RankingModel
from .DCN_Ranking import DCN

def use_retrieval_model(user_id):
    model = RetrievalModel([32])

    try:
        loaded = model.load('./recommender_engine/models/test-model')
        return loaded({'user_id': [user_id]})

    except Exception as e:
        model.fit_model()
        ids = model.predict_model(user_id)
        model.save('./recommender_engine/models/test-model')
        return ids

def use_ranking_model(user_id, ids):
    model = RankingModel([128, 128, 128])
    model.fit_model()
    model.evaluate_model()
    # model.predict_model(user_id, ids)


def use_dcn_ranking_model(user_id, ids):
    USE_CROSS_LAYER = True
    DEEP_LAYER_SIZES = []
    LAYER_SIZES_TOWERS = [32]
    PROJECTION_DIM = 20
    LEARNING_RATE = 0.1
    EPOCHS = 12

    model = DCN(
        use_cross_layer=USE_CROSS_LAYER, 
        deep_layer_sizes=DEEP_LAYER_SIZES, 
        projection_dim=PROJECTION_DIM,
        layer_sizes=LAYER_SIZES_TOWERS
    )
    model.fit_model(learning_rate=LEARNING_RATE, num_epoch=EPOCHS)
    model.evaluate_model()


if __name__ == "__main__":
    USER_ID = '26'
    # score, ids = use_retrieval_model(USER_ID)
    # use_ranking_model(USER_ID, [])

    use_dcn_ranking_model(USER_ID, [])