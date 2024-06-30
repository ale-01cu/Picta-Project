import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
from .tower.TowerModel import TowerModel
import typing as typ
from datetime import datetime
import re

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
        model_name: str,
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

        self.model_name = model_name
        self.epochs = None
        self.learning_rate = None

        self.metric_labels = (
            "factorized_top_k/top_1_categorical_accuracy",
            "factorized_top_k/top_5_categorical_accuracy",
            "factorized_top_k/top_10_categorical_accuracy",
            "factorized_top_k/top_50_categorical_accuracy",
            "factorized_top_k/top_100_categorical_accuracy"
        )

        self.evaluation_result = {
            "train": [],
            "test": []
        }

        self.hiperparams = (
            f"Towers Layers Sizes: {towers_layers_sizes}",
            f"Features Query: {[f for f in features_data_q.keys()]}",
            f"Features Candidate: {[f for f in features_data_c.keys()]}",
            f"Embedding Dimension: {embedding_dimension}",
            f"Shuffle: {shuffle}",
            f"Train Batch: {train_batch}",
            f"Test Batch: {test_batch}", 
            f"Candidate Batch: {candidates_batch}", 
            f"K Candidates: {k_candidates}" 
        )

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
        self.epochs = num_epochs
        self.learning_rate = learning_rate

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

        print("********** Train **********")
        for metric in self.metric_labels:
            output = f"Top-1 accuracy (train): {train_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["train"].append(output)

        test_accuracy = model.evaluate(
            self.cached_test, 
            return_dict=True,
            verbose=0
        )

        print("********** Test **********")
        for metric in self.metric_labels:
            output = f"Top-1 accuracy (test): {test_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["test"].append(output)



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
        def format_size(x):
            if x < 1000:
                return str(x)
            elif x < 1000000:
                return f"{x / 1000:.0f}K"
            elif x < 1000000000:
                return f"{x / 1000000:.0f}M"
            elif x < 1000000000000:
                return f"{x / 1000000000:.0f}B"
            else:
                return f"{x / 1000000000000:.0f}T"

        current_time = datetime.now()

        content = [
            f"Nombre: {self.model_name}",
            "\n ********** Hiperparametros **********",
            '\n'.join(self.hiperparams),
            f"Epochs: {self.epochs}",
            f"Learning Rate: {self.learning_rate}"
        ]

        for k, v in self.evaluation_result.items():
          content.append(f"\n ********** {k} **********")
          content.append("\n".join(metric for metric in v))

        content.append("\n ********** Parametros **********")
        total_params = 0
        for param in self.variables[:-1]:
            params = param.shape[0] * param.shape[1]
            total_params += params
            content.append(f"{param.name}: {params}")
        content.append(f"Total params: {total_params}")

        content = "\n".join(content)

        name = f"{self.model_name} ({format_size(total_params)}) {current_time}"
        name = re.sub(r'[^\w\s-]', '', name)  # remove invalid characters
        name = name.replace(' ', '_')  # replace spaces with underscores

        print(name)
        tf.saved_model.save(index, f"{path}/{name}")
        with open(f"{path}/{name}/Info.txt", "w") as f:
            f.write(f"{content}")

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'towers_layers_sizes': self.towers_layers_sizes,
            'vocabularies': self.vocabularies,
            'features_data_q': self.features_data_q,
            'features_data_c': self.features_data_c
        })
        return config

    def load_model(self, path: str) -> None:
        return tf.saved_model.load(path)