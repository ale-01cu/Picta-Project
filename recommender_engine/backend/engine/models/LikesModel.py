import tensorflow as tf
import tensorflow_recommenders as tfrs
import typing as typ
import numpy as np
from engine.models.tower.TowerModel import TowerModel
from datetime import datetime
import re
import os
import pickle
from engine.models.ModelConfig import ModelConfig
from engine.exceptions.Models.ModelInitializing import ModelInitializingException
from engine.exceptions.Models import (
    Evaluate,
    Fit,
    Indexing,
    Load,
    Save
)
dirname = os.path.dirname(__file__)

class LikesModel(tfrs.models.Model):
    
    def __init__(self,
        config: ModelConfig,
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        regularization_l2: float
        # model_name: str,
        # towers_layers_sizes: typ.List[int],
        # deep_layers_sizes: typ.List[int],
        # features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        # features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        # embedding_dimension: int = 32,
        # train: tf.data.Dataset,
        # val: tf.data.Dataset,
        # test: tf.data.Dataset,
        # shuffle: int = 20_000,
        # train_batch: int = 2048,
        # test_batch: int = 1024,
        # val_batch: int = 1024,
    ) -> None:
        try:
            print("Inicializando Modelo de Clasificacion de likes...")
            
            super().__init__()
            self.config = config
            self.model_name = config.model_name
            self.model_filename = None
            self.model_path = None
            self.data_train_path = None
            self.model_metadata_path = None
            self.epochs = None
            self.learning_rate = None
            self.vocabularies = vocabularies

            self.metric_labels = (
                "binary_accuracy",
                "loss"
            )

            self.evaluation_result = {
                "train": [],
                "test": []
            }

            self.hiperparams = (
                f"Towers Layers Sizes: {config.towers_layers_sizes}",
                f"Deep Layers Sizes: {config.deep_layers_sizes}",
                f"Features Query: {[f for f in config.features_data_q.keys()]}",
                f"Features Candidate: {[f for f in config.features_data_c.keys()]}",
                f"Embedding Dimension: {config.embedding_dimension}",
                # f"Shuffle: {shuffle}",
                # f"Train Batch: {train_batch}",
                # f"Test Batch: {test_batch}", 
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

            # ********* Red De Clasificacion Binaria **********
            self.likes_model = tf.keras.Sequential()

            # self.likes_model.add(tf.keras.layers.Dense(64, 
            #     kernel_initializer=tf.keras.initializers.RandomNormal(
            #         mean=0.0, stddev=0.05), input_shape=(None, 128)))
            
            for size in config.deep_layers_sizes:
                self.likes_model.add(tf.keras.layers.Dense(
                    size, activation='relu'))

            self.likes_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
            # ********* Red De Clasificacion Binaria **********


            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
            )

            # self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            #     loss=tf.keras.losses.MeanSquaredError(),
            #     metrics=[tf.keras.metrics.RootMeanSquaredError()],
            # )
        except:
            raise ModelInitializingException("Ranking")


    def call(self, inputs: typ.Dict[typ.Text, tf.Tensor]) -> tf.Tensor:
        user_embedding = self.query_model(inputs)
        pub_embedding = self.candidate_model(inputs)
        concated_embeddings = tf.concat(
            [user_embedding, pub_embedding], axis=1)

        return self.likes_model(concated_embeddings)
    

    def compute_loss(self, 
        features: typ.Dict[typ.Text, tf.Tensor], 
        training: bool = False
    ) -> tf.Tensor:
        
        labels = features.pop(self.config.target_column['new'])
        likes_predictions = self(features)
        # The task computes the loss and the metrics.

        return self.task(
            labels=labels, 
            predictions=likes_predictions
        )
    

    def fit_model(self, 
        cached_train,
        cached_val,
        callbacks=None, 
        # learning_rate: float = 0.1, 
        # num_epochs: int = 1,
        # use_multiprocessing: bool = False, 
        # workers: int = 1
    ) -> None:
        try:
            print(f'---------- Entrenando el modelo {self.model_name} ----------')
            self.epochs = self.config.num_epochs
            self.learning_rate = self.config.learning_rate
            model = self
            model.compile(optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.learning_rate))
            model.fit(
                cached_train, 
                epochs=self.config.num_epochs,
                validation_data=cached_val,
                callbacks=callbacks,
                use_multiprocessing=self.config.use_multiprocessing,
                workers=self.config.workers
            )
        except:
            raise Fit.FitException("Ranking")


    def evaluate_model(self, cached_test, cached_train) -> None:
        try:
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
        except:
            raise Evaluate.EvaluateException("Ranking")


    def predict_model(self, user_id: int, pubs_ids: list[str], candidates) -> None:
        model = self
        test_ratings = {}

        for id in pubs_ids:
            id = id
            model_input = self.get_row_as_dict(id, candidates)
            model_input['user_id'] = np.array([user_id])
            test_ratings[id] = model(model_input)

        print("Ratings:")
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{title}: {score}")


    def get_row_as_dict(self, id: str, candidates) -> typ.Dict:
        id = int(id)
        # id = str(id)
        # Filtrar el DataFrame para solo la fila donde el id coincide con el proporcionado
        df_filtered = candidates[candidates['id'] == id]
        # Convertir la primera (y única) fila del DataFrame filtrado a un diccionario
        row_as_dict = df_filtered.iloc[0].to_dict()
        
        # Convertir cada valor del diccionario a un numpy.array
        for key in row_as_dict:
            row_as_dict[key] = np.array([row_as_dict[key]])
        
        return row_as_dict
    
    # def get_config(self):
    #     config = {
    #         'model_name': self.model_name,
    #         'towers_layers_sizes': self.query_model.layer_sizes,
    #         'deep_layers_sizes': [layer.units for layer in self.likes_model.layers if isinstance(layer, tf.keras.layers.Dense)],
    #         'vocabularies': {key: value.numpy() for key, value in self.query_model.vocabularies.items()},
    #         'features_data_q': {key: {k: v.numpy() for k, v in value.items()} for key, value in self.query_model.features_data.items()},
    #         'features_data_c': {key: {k: v.numpy() for k, v in value.items()} for key, value in self.candidate_model.features_data.items()},
    #         'embedding_dimension': self.query_model.embedding_dimension,
    #         'shuffle': self.cached_train.shuffle_buffer_size,
    #         'train_batch': self.cached_train.batch_size,
    #         'test_batch': self.cached_test.batch_size,
    #         'val_batch': self.cached_val.batch_size,
    #     }
    #     return config
    
    def save_model(self, path: str, dataset: tf.data.Dataset) -> None:
        try:
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

            self.model_path = f"{path}/{name}"
            self.model_filename = name

            # tf.saved_model.save(self, f"{path}/{name}")

            # print("Salvando el modelo...")
            # self.save(f"{path}/{name}")

            print("Salvando los pesos...")
            self.save_weights(
                f"{self.model_path}/model/pesos.tf", 
                save_format='tf'
            )

            print("Salvando el Modelo")
            tf.saved_model.save(self, f"{self.model_path}/service")

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
        except:
            raise Save.SaveException("Ranking")

    def load_model(self, path: str, cached_train, cached_test) -> None:
        try:
            print("Cargando los pesos...")
            status = self.load_weights(os.path.join(path, "model/pesos.tf"))
        except:
            raise Load.LoadException("Ranking")
        # status.expect_partial()

        # print("Compilando...")
        # self.compile(optimizer=tf.keras.optimizers.Adam(
        #     learning_rate=self.config.learning_rate)
        # )
        # print("Inicializando...")
        # cached_train.map(lambda x: self(x))
        
        # print("Evaluando...")
        # self.evaluate_model(
        #     cached_test=cached_test,
        #     cached_train=cached_train
        # )
        