import tensorflow as tf
import tensorflow_recommenders as tfrs
import typing as typ
import numpy as np
from recommender_engine.tower.TowerModel import TowerModel
from datetime import datetime
import re
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)

class LikesModel(tfrs.models.Model):
    
    def __init__(self,
        model_name: str,
        towers_layers_sizes: typ.List[int],
        deep_layers_sizes: typ.List[int],
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        train: tf.data.Dataset,
        val: tf.data.Dataset,
        test: tf.data.Dataset,
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        val_batch: int = 1024,
        embedding_dimension: int = 32,
    ) -> None:
        super().__init__()

        self.cached_train = train.shuffle(shuffle)\
            .batch(train_batch).cache()
        self.cached_test = test.batch(test_batch).cache()
        self.cached_val = val.batch(val_batch).cache()


        self.model_name = model_name
        self.model_filename = None
        self.epochs = None
        self.learning_rate = None

        self.metric_labels = (
            "binary_accuracy",
            "loss"
        )

        self.evaluation_result = {
            "train": [],
            "test": []
        }

        self.hiperparams = (
            f"Towers Layers Sizes: {towers_layers_sizes}",
            f"Deep Layers Sizes: {deep_layers_sizes}",
            f"Features Query: {[f for f in features_data_q.keys()]}",
            f"Features Candidate: {[f for f in features_data_c.keys()]}",
            f"Embedding Dimension: {embedding_dimension}",
            f"Shuffle: {shuffle}",
            f"Train Batch: {train_batch}",
            f"Test Batch: {test_batch}", 
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

        # ********* Red De Clasificacion Binaria **********
        self.likes_model = tf.keras.Sequential()

        # self.likes_model.add(tf.keras.layers.Dense(64, 
        #     kernel_initializer=tf.keras.initializers.RandomNormal(
        #         mean=0.0, stddev=0.05), input_shape=(None, 128)))
        
        for size in deep_layers_sizes:
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
        
        labels = features.pop("like_dislike")
        likes_predictions = self(features)
        # The task computes the loss and the metrics.

        return self.task(
            labels=labels, 
            predictions=likes_predictions
        )
    

    def fit_model(self, callbacks=None, 
        learning_rate: float = 0.1, 
        num_epochs: int = 1,
        use_multiprocessing: bool = False, 
        workers: int = 1
    ) -> None:
        
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate))
        model.fit(
            self.cached_train, 
            epochs=num_epochs,
            validation_data=self.cached_val,
            callbacks=callbacks,
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
            output = f"{metric} (train): {train_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["train"].append(output)

        test_accuracy = model.evaluate(
            self.cached_test, 
            return_dict=True,
            verbose=0
        )

        print("********** Test **********")
        for metric in self.metric_labels:
            output = f"{metric} (test): {test_accuracy[metric]:.2f}."
            print(output)
            self.evaluation_result["test"].append(output)


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
    
    def save_model(self, path: str) -> None:
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
            params = param.shape[0] if len(param.shape) == 1 else param.shape[0] * param.shape[1]
            total_params += params
            content.append(f"{param.name}: {params}")
        content.append(f"Total params: {total_params}")

        content = "\n".join(content)

        name = f"{self.model_name} ({format_size(total_params)}) {current_time}"
        name = re.sub(r'[^\w\s-]', '', name)  # remove invalid characters
        name = name.replace(' ', '_')  # replace spaces with underscores

        self.model_filename = name
        tf.saved_model.save(self, f"{path}/{name}")
        with open(f"{path}/{name}/Info.txt", "w") as f:
            f.write(f"{content}")

import os
import pandas as pd
from .data.DataPipelineBase import DataPipelineBase


dirname = os.path.dirname(__file__)
pubs_path = os.path.join(dirname, "../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv")

pubs_df = pd.read_csv(pubs_path)
pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
pubs_df['nombre'] = pubs_df['nombre'].astype(str)
pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))

if __name__ == "__main__":
    views_path = os.path.join(dirname, "../datasets/likes.csv")
    features = ['usuario_id', 'id', 'like_dislike']
    print('Cargando la data...')
    pipeline = DataPipelineBase(dataframe_path=views_path)
    pipeline.dataframe = pipeline.dataframe.drop(['id'], axis=1)
    # pipeline.dataframe = pipeline.dataframe[: 100_000]

    pipeline.dataframe['like_dislike'] = pipeline.dataframe['valor'].map({True: 1, False: 0})

    df = pipeline.merge_data(
        df_to_merge=pubs_df, 
        left_on='publicacion_id',
        right_on='id',
        output_features=features
    )
    ds = pipeline.convert_to_tf_dataset(df)
    print('Construyendo vocabulario...')
    vocabularies = pipeline.build_vocabularies(
        features=features, ds=ds, batch=1_000)
    
    total, train_Length, val_length, test_length = pipeline.get_lengths(ds)

    train, val, test = pipeline.split_into_train_and_test(
        ds=ds,
        shuffle=100_000,
        train_length=train_Length,
        val_length=val_length,
        test_length=test_length,
        seed=42
    )

    model = LikesModel(
        model_name="Likes Lite",
        towers_layers_sizes=[],
        deep_layers_sizes=[],
        vocabularies=vocabularies,
        features_data_q={
            'usuario_id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous.CategoricalContinuous, 'w': 0.3 }    
        },
        features_data_c={ 
            'id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'nombre': { 'dtype': StringText.StringText, 'w': 0.2 },
            # 'descripcion': { 'dtype': StringText.StringText, 'w': 0.1 }
        },
        embedding_dimension=64, 
        train=train, 
        test=test, 
        val=val,
        shuffle=100_000, 
        train_batch=8192, 
        test_batch=4096, 
    )

    model.fit_model(
        learning_rate=0.1,
        num_epochs=1,
        use_multiprocessing=True,
        workers=4   
    )

    model.evaluate_model()
    model.save_model("models")
