import tensorflow_recommenders as tfrs
import tensorflow as tf
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from .data_pipeline import pubs_ds, train, test, total
import numpy as np

class RetrievalModel(tfrs.models.Model):
    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        # self.task = tfrs.tasks.Retrieval(
        #     metrics=tfrs.metrics.FactorizedTopK(
        #         candidates=pubs_ds.batch(128).map(self.candidate_model),
        #     ),
        # )

        self.task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(
                candidates=pubs_ds.batch(128).map(lambda c: (c["id"], self.candidate_model(c)))
            )
        )


        self.cached_train = train.shuffle(total).batch(2048).cache()
        self.cached_test = test.batch(1024).cache()

        self.num_epoch = 3


    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features)
        pubs_embeddings = self.candidate_model(features)
        return self.task(query_embeddings, pubs_embeddings)


    def fit_model(self):
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        model.fit(self.cached_train, epochs=self.num_epochs)
        return model
    

    def evaluate_model(self):
        model = self
        # model.evaluate(cached_test, return_dict=True)

        train_accuracy = model.evaluate(
            self.cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
        test_accuracy = model.evaluate(
            self.cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

        print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
        print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")


    def predict_model(self, user_id):
        print('--------- Prediciendo con el modelo ----------')
        model = self
        brute_force = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
        brute_force.index_from_dataset(
            pubs_ds.batch(128).map(lambda x: (x['id'], model.candidate_model(x)))
        )
        score, titles = brute_force({'user_id': np.array([user_id])}, k=10)
        print(f"Top recommendations: {titles[0]}")
        print(f"Score: {score}")
        return score, titles[0]


    def save(self, path):
        tf.saved_model.save(self, path)


    def load(self, path):
        return tf.saved_model.load(path)