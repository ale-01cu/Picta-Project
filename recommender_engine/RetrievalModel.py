import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
from .tower.TowerModel import TowerModel
import typing as typ

class RetrievalModel(tfrs.models.Model):
    """
    
    Los datos para entrenar este modelo son son pares 
    de tipo usuario - item osea en el caso de las publicaciones
    tengo que pasarle un historial de clicks que ha dado cada usuario 
    a cada publicacion
    
    ejemplo:

    En cualquier pagina de la aplicacion el usuario x dio click en la pelicula z
    
    """
    def __init__(self, 
        towers_layers_sizes: typ.List[int],
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        train: tf.data.Dataset, 
        test: tf.data.Dataset,
        val: tf.data.Dataset,
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
        self.cached_train = train.shuffle(self.shuffle)\
            .batch(self.train_batch).cache()
        self.cached_test = test.batch(self.test_batch).cache()
        self.cached_val = val.batch(self.test_batch).cache()

        self.query_model = TowerModel(
            layer_sizes=towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=features_data_q,
            embedding_dimension=embedding_dimension
        )
        self.candidate_model = TowerModel(
            layer_sizes=towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=features_data_c,
            embedding_dimension=embedding_dimension,
        )

        self.task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(
                candidates=self.candidates.batch(
                    self.candidates_batch).map(self.candidate_model)
            )
        )


    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features)
        pubs_embeddings = self.candidate_model(features)
        return self.task(query_embeddings, pubs_embeddings)


    def fit_model(self, 
        learning_rate: float = 0.1, 
        num_epochs: int = 1, 
        use_multiprocessing: bool = False, 
        workers: int = 1
    ) -> None:
        
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=learning_rate))
        model.fit(
            self.cached_train, 
            validation_data=self.cached_val,
            epochs=num_epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers
        )
    

    def evaluate_model(self) -> None:
        model = self
        model.evaluate(self.cached_test, return_dict=True)

        train_accuracy = model.evaluate(
            self.cached_train, 
            return_dict=True,
            verbose=0
        )

        train_top_1 = train_accuracy["factorized_top_k/top_1_categorical_accuracy"]
        train_top_5 = train_accuracy["factorized_top_k/top_5_categorical_accuracy"]
        train_top_10 = train_accuracy["factorized_top_k/top_10_categorical_accuracy"]
        train_top_50 = train_accuracy["factorized_top_k/top_50_categorical_accuracy"]
        train_top_100 = train_accuracy["factorized_top_k/top_100_categorical_accuracy"]

        test_accuracy = model.evaluate(
            self.cached_test, 
            return_dict=True,
            verbose=0
        )

        test_top_1 = test_accuracy["factorized_top_k/top_1_categorical_accuracy"]
        test_top_5 = test_accuracy["factorized_top_k/top_5_categorical_accuracy"]
        test_top_10 = test_accuracy["factorized_top_k/top_10_categorical_accuracy"]
        test_top_50 = test_accuracy["factorized_top_k/top_50_categorical_accuracy"]
        test_top_100 = test_accuracy["factorized_top_k/top_100_categorical_accuracy"]

        print('**********')
        print(f"Top-1 accuracy (train): {train_top_1:.2f}.")
        print(f"Top-5 accuracy (train): {train_top_5:.2f}.")
        print(f"Top-10 accuracy (train): {train_top_10:.2f}.")
        print(f"Top-50 accuracy (train): {train_top_50:.2f}.")
        print(f"Top-100 accuracy (train): {train_top_100:.2f}.")

        print('**********')
        print(f"Top-1 accuracy (test): {test_top_1:.2f}.")
        print(f"Top-5 accuracy (test): {test_top_5:.2f}.")
        print(f"Top-10 accuracy (test): {test_top_10:.2f}.")
        print(f"Top-50 accuracy (test): {test_top_50:.2f}.")
        print(f"Top-100 accuracy (test): {test_top_100:.2f}.")


    def index_model(self):
        index = tfrs.layers.factorized_top_k.BruteForce(
            self.query_model, 
            k=self.k_candidates
        )

        index.index_from_dataset(
            self.candidates.batch(self.candidates_batch).map(
                lambda x: (x['id'], self.candidate_model(x)))
        )

        return index


    def predict_model(self, index: tfrs.layers.factorized_top_k.BruteForce, user_id: int) -> tuple[tf.Tensor, tf.Tensor]:
        print('--------- Prediciendo con el modelo ----------')
        score, titles = index(
            {'user_id': np.array([user_id])}, 
            k=self.k_candidates
        )
        
        return score, titles[0]


    def save_model(self, index: tfrs.layers.factorized_top_k.BruteForce, path: str) -> None:
        tf.saved_model.save(index, path)


    def load_model(self, path: str) -> None:
        return tf.saved_model.load(path)