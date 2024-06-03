import tensorflow_recommenders as tfrs
import tensorflow as tf
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from tensorflow.python.types.core import Tensor
import numpy as np
# from .data_pipeline import unique_pubs_ids
from typing import Dict, Text

class SecuntialRetrievalModel(tfrs.models.Model):
    """
    
    Los datos para entrenar este modelo son son pares 
    de tipo:
    cantidad de peliculas que le ha dado click el usuario - la proxima pelicula que vera
    las cantidades de peliculas anteriores se define dependiendo del tamano del dataset
    y la proxima pelicula siempre es una.
    
    Aqui se tiene en cuenta el orden de las peliculas anteriormente vistas por tanto
    se necesita una caracteristica de tiempo.

    ejemplo:

    Cantidad x de peliculas que ha visto el usuario anteriormente y la siguiente pelicula que vio
    y las peliculas anteriores todas tienen una marca de tiempo.

    
    """
    def __init__(self, 
        layer_sizes: list[int], 
        train: tf.data.Dataset, 
        test: tf.data.Dataset,
        candidates: tf.data.Dataset,
        vocabularies: Dict[Text, Dict[Text, tf.Tensor]],
        features_names_q: list[str],
        features_names_c: list[str],
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
        self.embedding_dimension = embedding_dimension
        self.vocabularies = vocabularies
        self.features_names_q = features_names_q
        self.features_names_c = features_names_c

        self.candidates = candidates
        self.cached_train = train.shuffle(self.shuffle).batch(
            self.train_batch).cache()
        self.cached_test = test.batch(self.test_batch).cache()


        self.query_model = CandidateModel(
            vocabularies=vocabularies,
            features_names=features_names_c,
            layer_sizes=layer_sizes, 
            embedding_dimension=embedding_dimension,
            aditional_layers=[tf.keras.layers.GRU(self.embedding_dimension)]
        )

        self.candidate_model = CandidateModel(
            vocabularies=vocabularies,
            features_names=features_names_c,
            layer_sizes=layer_sizes, 
            embedding_dimension=embedding_dimension
        )


        self.index = tfrs.layers.factorized_top_k.BruteForce(
            self.query_model, k=self.k_candidates)
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
        query_embeddings = tf.reshape(query_embeddings, [-1, 34])
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


    def predict_model(self, publication_id: int) -> tuple[Tensor, Tensor]:
        print('--------- Prediciendo con el modelo ----------')
        model = self
        brute_force = self.index

        brute_force.index_from_dataset(
            self.candidates.batch(self.candidates_batch).map(
                lambda x: (x['id'], model.candidate_model(x)))
        )

        score, titles = brute_force(
            {'publication_id': np.array([publication_id])}, 
            k=self.k_candidates
        )
        
        return score, titles[0]


    def save_model(self, path: str) -> None:
        tf.saved_model.save(self.index, path)


    def load_model(self, path: str) -> None:
        return tf.saved_model.load(path)