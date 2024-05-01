import tensorflow_recommenders as tfrs
import tensorflow as tf
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from tensorflow.python.types.core import Tensor
import numpy as np

class RetrievalModel(tfrs.models.Model):
    def __init__(self, 
        layer_sizes: list[int], 
        train: tf.data.Dataset, 
        test: tf.data.Dataset,
        candidates: tf.data.Dataset,
        embedding_dimension: int = 32, 
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        candidates_batch: int = 128,
        k_candidates: int = 10,
    ) -> None:

        super().__init__()
        self.shuffle = shuffle
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.candidates_batch = candidates_batch
        self.k_candidates = k_candidates

        self.candidates = candidates
        self.cached_train = train.shuffle(self.shuffle).batch(self.train_batch).cache()
        self.cached_test = test.batch(self.test_batch).cache()

        self.query_model = QueryModel(
            layer_sizes=layer_sizes, 
            embedding_dimension=embedding_dimension
        )

        self.candidate_model = CandidateModel(
            layer_sizes=layer_sizes,
            embedding_dimension=embedding_dimension
        )

        self.index = tfrs.layers.factorized_top_k.BruteForce(
            self.query_model, 
            k=self.k_candidates
        )
        # self.task = tfrs.tasks.Retrieval(
        #     metrics=tfrs.metrics.FactorizedTopK(
        #         candidates=pubs_ds.batch(128).map(self.candidate_model),
        #     ),
        # )

        self.task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(
                candidates=self.candidates.batch(self.candidates_batch).map(
                    lambda c: (c["id"], self.candidate_model(c)))
            )
        )


    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features)
        pubs_embeddings = self.candidate_model(features)
        return self.task(query_embeddings, pubs_embeddings)


    def fit_model(self, learning_rate: float = 0.1, num_epochs: int = 1) -> None:
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate))
        model.fit(self.cached_train, epochs=num_epochs)
    

    def evaluate_model(self) -> None:
        model = self
        model.evaluate(self.cached_test, return_dict=True)

        train_accuracy = model.evaluate(
            self.cached_train, 
            return_dict=True
        )["factorized_top_k/top_100_categorical_accuracy"]
        test_accuracy = model.evaluate(
            self.cached_test, 
            return_dict=True
        )["factorized_top_k/top_100_categorical_accuracy"]

        print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
        print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")


    def predict_model(self, user_id: int) -> tuple[Tensor, Tensor]:
        print('--------- Prediciendo con el modelo ----------')
        model = self
        brute_force = self.index

        brute_force.index_from_dataset(
            self.candidates.batch(self.candidates_batch).map(
                lambda x: (x['id'], model.candidate_model(x)))
        )

        score, titles = brute_force(
            {'user_id': np.array([user_id])}, 
            k=self.k_candidates
        )
        
        return score, titles[0]


    def save_model(self, path: str) -> None:
        tf.saved_model.save(self.index, path)


    def load_model(self, path: str) -> None:
        return tf.saved_model.load(path)