import tensorflow as tf
import tensorflow_recommenders as tfrs
import typing as typ
from typing import Text
import typing as typ
import numpy as np
from .tower.TowerModel import TowerModel
import pandas as pd
from recommender_engine.data.featurestypes import (
    StringText, CategoricalContinuous, CategoricalString, CategoricalInteger)
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from .data.DataPipelineBase import DataPipelineBase
from keras.preprocessing.text import Tokenizer
from recommender_engine.utils.utils import update_json
import re
from datetime import datetime

class PositiveModel(tfrs.models.Model):
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
        deep_layer_sizes: list[int],
        vocabularies: typ.Dict[typ.Text, typ.Dict[typ.Text, tf.Tensor]],
        features_data_q: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        features_data_c: typ.Dict[typ.Text, typ.Dict[typ.Text, object]],
        train: tf.data.Dataset,
        test: tf.data.Dataset,
        val: tf.data.Dataset,
        embedding_dimension: int = 32,
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        val_batch: int = 1024,

    ) -> None:

        super().__init__()
        self.shuffle = shuffle
        self.train_batch = train_batch
        self.test_batch = test_batch


        self.cached_train = train.shuffle(self.shuffle)\
            .batch(self.train_batch).cache()
        self.cached_test = test.batch(self.test_batch).cache()
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
            f"Deep Layers Sizes: {deep_layer_sizes}",
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

        self.positive_layers = tf.keras.Sequential()
        # self.positive_layers.add(tf.keras.layers.Dropout(0.5))

        # self.positive_layers.add(tf.keras.layers.Dropout(0.2))
        self.positive_layers.add(tf.keras.layers.Dense(64,
          activation="relu", 
          kernel_regularizer=tf.keras.regularizers.l2(0.01), 
          activity_regularizer=tf.keras.regularizers.l2(0.01)
        ))

        for size in deep_layer_sizes:
          self.positive_layers.add(tf.keras.layers.Dense(size, activation="relu"))
        #   self.positive_layers.add(tf.keras.layers.Dropout(0.01))

        self.positive_layers.add(tf.keras.layers.Dense(units=3, activation='sigmoid'))


        # self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        #     loss=tf.keras.losses.CategoricalCrossentropy(),
        #     metrics=[tf.keras.metrics.CategoricalAccuracy()],
        # )
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

    def call(self, inputs: dict[Text, tf.Tensor]) -> tf.Tensor:
        user_embedding = self.query_model(inputs)
        pub_embedding = self.candidate_model(inputs)

        return self.positive_layers(tf.concat(
            [user_embedding, pub_embedding], axis=1))


    def compute_loss(self, features, training=False):
        labels = features.pop("category")
        predictions = self(features)

        return self.task(
            labels=labels,
            predictions=predictions
        )


    def summary_model(self):
      self.query_model.summary()
      self.candidate_model.summary()
      self.positive_layers.summary()
      self.summary()


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

        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate))
        history = model.fit(
            self.cached_train, 
            validation_data=self.cached_val,
            epochs=num_epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers
        )

        return history

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



    def predict_model(self, user_id: int) -> tuple[tf.Tensor, tf.Tensor]:
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

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    positive_data_path = os.path.join(dirname, "../datasets/positive_data.csv")
    pubs_path = os.path.join(dirname, "../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv")
    pubs_df = pd.read_csv(pubs_path)
    pubs_df['descripcion'] = pubs_df['descripcion'].astype(str)
    pubs_df['nombre'] = pubs_df['nombre'].astype(str)

    pubs_ds = tf.data.Dataset.from_tensor_slices(dict(pubs_df))

    # x = ds.map(lambda x: {
    #     'user_id': x['user_id'],
    #     'publication_id': x['publication_id'],
    #     'timestamp': x['timestamp']
    # })
    # # y = ds.map(lambda x: x['category'])

    # # x_train = x.take(80_000)
    # # y_train = y.take(80_000)

    # # x_test = x.skip(80_000).take(20_000)
    # # y_test = y.skip(80_000).take(20_000)

    def map_to_one_hot(elements: list):
        labels = elements
        unique_labels = np.unique(labels)

        # Supongamos que 'y' son tus etiquetas de salida
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(unique_labels)

        onehot_encoded = to_categorical(integer_encoded)

        map = {
            elem: one_hot.tolist()
            for elem, one_hot in
            zip(unique_labels, onehot_encoded)
        }
        return np.array([map[elem] for elem in elements])
    

    features = ['usuario_id', 'id', 'category']
    pipeline = DataPipelineBase(dataframe_path=positive_data_path)
    pipeline.dataframe = pipeline.dataframe[: 100_000]
    
    
    df = pipeline.merge_data(
        df_to_merge=pubs_df,
        left_on='publicacion_id',
        right_on='id',
        output_features=features
    )

    data = dict(df)
    one_hot =  map_to_one_hot(df['category'].tolist())
    data['category'] = one_hot


    ds = pipeline.convert_to_tf_dataset(data)
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


    model = PositiveModel(
        model_name="Positive Ranking Lite",
        towers_layers_sizes=[],
        deep_layer_sizes=[],
        vocabularies=vocabularies,
        features_data_q={
            'usuario_id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'timestamp': { 'dtype': CategoricalContinuous, 'w': 0.2 }
        },
        features_data_c={
            'id': { 'dtype': CategoricalInteger.CategoricalInteger, 'w': 1 },
            # 'nombre': { 'dtype': StringText, 'w': 0.1 }
        },
        embedding_dimension=64,
        train=train,
        test=test,
        val=val,
        shuffle=100_000,
        train_batch=4096,
        test_batch=1024,
    )

    history = model.fit_model(learning_rate=0.0001, num_epochs=200)
    print('Evaluando...')
    model.evaluate_model()
    model.save_model(os.path.join(dirname, "models"))
    update_json("./models/info.json", {
        "positive_model_name": model.model_filename
    })