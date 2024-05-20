import tensorflow as tf
import tensorflow_recommenders as tfrs
import typing as typ
import numpy as np
from .tower.TowerModel import TowerModel

class likesModel(tfrs.models.Model):
    
    def __init__(self,
        towers_layers_sizes: typ.List[int],
        likes_layers_sizes: typ.List[int],
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
        
        for size in likes_layers_sizes:
            self.likes_model.add(tf.keras.layers.Dense(
                size, activation='relu'))

        self.likes_model.add()
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
    

    def fit_model(self, callbacks, 
        learning_rate: float = 0.1, 
        num_epochs: int = 1
    ) -> None:
        
        print('---------- Entrenando el modelo ----------')
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate))
        model.fit(
            self.cached_train, 
            epochs=num_epochs,
            validation_data=self.cached_val,
            callbacks=callbacks
        )


    def evaluate_model(self) -> None:
        self.evaluate(self.cached_test, return_dict=True)


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