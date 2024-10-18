import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
from engine.models.tower.TowerModel import TowerModel
import typing as typ
from datetime import datetime
import re
import os
import pickle
from engine.models.ModelConfig import ModelConfig
from engine.data import FeaturesTypes
from engine.data.DataPipeline import DataPipeline
dirname = os.path.dirname(__file__)

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
        # model_name: str,
        # towers_layers_sizes: typ.List[int],
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        regularization_l2: float,
        # features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        # features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        # # train: tf.data.Dataset, 
        # # test: tf.data.Dataset,
        # # val: tf.data.Dataset,
        candidates: tf.data.Dataset,
        # embedding_dimension: int = 32, 
        # # shuffle: int = 20_000,
        # # train_batch: int = 2048,
        # # test_batch: int = 1024,
        # candidates_batch: int = 128,
        # k_candidates: int = 10,
        config: ModelConfig
    ) -> None:
        print("Inicializando Modelo de Recuperacion...")
        super().__init__()
        # self.shuffle = shuffle
        self.config = config
        self.candidates_batch = config.candidates_batch
        self.k_candidates = config.k_candidates
        self.candidates = candidates
        self.vocabularies = vocabularies

        # self.cached_train = train.shuffle(self.shuffle)\
        #     .batch(self.train_batch).cache()
        # self.cached_test = test.batch(self.test_batch).cache()
        # self.cached_val = val.batch(self.test_batch).cache()

        self.model_name = config.model_name
        self.model_filename = None
        self.model_path = None
        self.data_train_path = None
        self.model_metadata_path = None
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
            f"Towers Layers Sizes: {config.towers_layers_sizes}",
            f"Features Query: {[f for f in config.features_data_q.keys()]}",
            f"Features Candidate: {[f for f in config.features_data_c.keys()]}",
            f"Embedding Dimension: {config.embedding_dimension}",
            # f"Shuffle: {shuffle}",
            # f"Train Batch: {train_batch}",
            # f"Test Batch: {test_batch}", 
            f"Candidate Batch: {config.candidates_batch}", 
            f"K Candidates: {config.k_candidates}" 
        )

        self.query_model = TowerModel(
            layer_sizes=config.towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=config.features_data_q,
            embedding_dimension=config.embedding_dimension,
            regularization_l2=regularization_l2
        )
        self.candidate_model = TowerModel(
            layer_sizes=config.towers_layers_sizes,
            vocabularies=vocabularies,
            features_data=config.features_data_c,
            embedding_dimension=config.embedding_dimension,
            regularization_l2=regularization_l2
        )

        self.task = tfrs.tasks.Retrieval(
            metrics= tfrs.metrics.FactorizedTopK(
                candidates=self.candidates.batch(
                    self.candidates_batch).map(self.candidate_model)
            )
        )

    def call(self, inputs):
        user_embedding = self.query_model(inputs)
        pub_embedding = self.candidate_model(inputs)

        return tf.concat(
            [user_embedding, pub_embedding], 
            axis=1
        )


    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features)
        pubs_embeddings = self.candidate_model(features)
        return self.task(query_embeddings, pubs_embeddings)
        

    def fit_model(self, 
        # train: tf.data.Dataset, 
        # val: tf.data.Dataset,
        # train_batch: int = 2048,
        # val_batch: int = 1024,
        # shuffle: int = 20_000,
        cached_train,
        cached_val,
        # learning_rate: float = 0.1, 
        # num_epochs: int = 1, 
        # use_multiprocessing: bool = False, 
        # workers: int = 1,
    ) -> None:
        print(f'---------- Entrenando el modelo {self.model_name} ----------')
        model = self
        self.epochs = self.config.num_epochs
        self.learning_rate = self.config.learning_rate

        model.compile(optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=self.config.learning_rate))
        model.fit(
            cached_train, 
            validation_data=cached_val,
            epochs=self.config.num_epochs,
            use_multiprocessing=self.config.use_multiprocessing,
            workers=self.config.workers
        )
    

    def evaluate_model(self, cached_test, cached_train) -> None:
        model = self
        model.evaluate(cached_test, return_dict=True)

        train_accuracy = model.evaluate(
            cached_train, 
            return_dict=True,
            verbose=0
        )

        print("********** Train **********")
        for metric in self.metric_labels:
            output = f"{metric} (train): {train_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["train"].append(output)

        test_accuracy = model.evaluate(
            cached_test, 
            return_dict=True,
            verbose=0
        )

        print("********** Test **********")
        for metric in self.metric_labels:
            output = f"{metric} (test): {test_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["test"].append(output)


    def index_model(self) -> tfrs.layers.factorized_top_k.BruteForce:
        print("Indexando...")

        data_test = {}

        for key, value in self.config.features_data_q.items():
            new_value = None
            feature_type = value['dtype']

            if(feature_type == FeaturesTypes.CategoricalString):
                new_value = np.array(["test"])
            elif(feature_type == FeaturesTypes.CategoricalInteger):
                new_value = np.array([0])
            elif(feature_type == FeaturesTypes.CategoricalContinuous):
                new_value = np.array([0])
            elif(feature_type == FeaturesTypes.StringText):
                new_value = np.array(["test"])

            data_test[key] = new_value

        self.query_model(data_test)

        index = tfrs.layers.factorized_top_k.BruteForce(
            self.query_model, 
            k=self.k_candidates
        )

        index.index_from_dataset(
            self.candidates.batch(self.candidates_batch).map(
                lambda x: (x['id'], self.candidate_model(x)))
        )

        data_test = [
            {
                "usuario_id": np.array([320]),
                "edad": np.array([32])
            },
            {
                "usuario_id": np.array([161]),
                "edad": np.array([37])
            },
            {
                "usuario_id": np.array([3040]),
                "edad": np.array([51])
            },
            {
                "usuario_id": np.array([8097]),
                "edad": np.array([45])
            },
            {
                "usuario_id": np.array([9364]),
                "edad": np.array([22])
            },
        ]

        # index.build(input_shape=(None, self.config.embedding_dimension))

        data_test = {}

        for key, value in self.config.features_data_q.items():
           new_value = None
           feature_type = value['dtype']

           if(feature_type == FeaturesTypes.CategoricalString):
               new_value = np.array(["test"])
           elif(feature_type == FeaturesTypes.CategoricalInteger):
               new_value = np.array([0])
           elif(feature_type == FeaturesTypes.CategoricalContinuous):
               new_value = np.array([0])
           elif(feature_type == FeaturesTypes.StringText):
               new_value = np.array(["test"])

           data_test[key] = new_value

        score, titles = index(data_test, self.k_candidates)
        ids = [id for id, score in zip(titles.numpy()[0], score.numpy()[0])]
        print(ids[: 10])

        #for data in data_test:
        #    print(data)
        #    score, titles = index(data, self.k_candidates)
        #    ids = [id for id, score in zip(titles.numpy()[0], score.numpy()[0])]
        #    print(ids[: 10])
        #    print("")

        return index


    def predict_model(self, index: tfrs.layers.factorized_top_k.BruteForce, user_id: int) -> tuple[tf.Tensor, tf.Tensor]:
        print('--------- Prediciendo con el modelo ----------')
        score, titles = index(
            {'user_id': np.array([user_id])}, 
            k=self.k_candidates
        )
        
        return score, titles[0]


    def save_model(self, path: str, dataset: tf.data.Dataset) -> None:
        path = os.path.join(dirname, f"../{path}")

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
            if hasattr(param, 'shape'):
                params = 1
                for p in param.shape: params *= p 
                total_params += params
                content.append(f"{param.name}: {params}")
        content.append(f"Total params: {total_params}")

        content = "\n".join(content)

        name = f"{self.model_name} ({format_size(total_params)}) {current_time}"
        name = re.sub(r'[^\w\s-]', '', name)  # remove invalid characters
        name = name.replace(' ', '_')  # replace spaces with underscores

        model = self
        index = self.index_model()

        os.makedirs(f"{path}/{name}", exist_ok=True)
        #Problemas aqui para cargarlo 

        print(content)
        self.model_path = f"{path}/{name}"
        self.model_filename = name
        
        print("Salvando los pesos...")
        model.save_weights(f"{path}/{name}/model/pesos.tf", save_format='tf')
        # Problemas aqui para guardarlo 
        # tf.saved_model.save(model, f"{path}/{name}/model")
        print("Salvando los indices...")
        tf.saved_model.save(index, f"{path}/{name}/index")

        print("Salvando los datos de entrenamiento...")
        data_path = f"{path}/{name}/data"
        self.data_train_path = data_path
        dataset.save(data_path)
        with open(f"{data_path}/vocabularies.pkl", 'wb') as f:
            pickle.dump(self.vocabularies, f)

        self.model_metadata_path = f"{self.model_path}/hiperparams.json"
        self.config.save_as_json(self.model_metadata_path)
        with open(f"{path}/{name}/Info.txt", "w") as f:
            f.write(f"{content}")


    def load_model(self, path: str, cached_train, cached_test) -> None:
        
        print("Cargando los pesos...")
        self.load_weights(os.path.join(path, "model/pesos.tf"))
        print("Compilando...")
        self.compile(optimizer=tf.keras.optimizers.Adagrad(
            learning_rate=0.1)
        )
        print("Inicializando...")
        cached_train.map(lambda x: self(x))
        print("Evaluando...")
        self.evaluate_model(
            cached_test=cached_test,
            cached_train=cached_train
        )

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'vocabularies': self.vocabularies,
    #         'regularization_l2': self.query_model.embedding_model.get_config()['regularization_l2'],
    #         'candidates_batch': self.candidates_batch,
    #         'k_candidates': self.k_candidates,
    #         'config': self.config.get_config(),
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     # Deserializar la configuración del objeto ModelConfig
    #     config['config'] = ModelConfig.from_config(config['config'])
        
    #     return cls(**config)