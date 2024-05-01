import tensorflow_recommenders as tfrs
import tensorflow as tf
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel
from typing import Dict

class DCN(tfrs.Model):

    def __init__(self, 
        deep_layer_sizes_towers: list[int],
        deep_layer_sizes_ranking: list[int], 
        train: tf.data.Dataset,
        test: tf.data.Dataset,
        shuffle: int = 20_000,
        train_batch: int = 2048,
        test_batch: int = 1024,
        embedding_dimension: int = 32,
        use_cross_layer: bool = False, 
        projection_dim: int | None = None, 
    ) -> None:
        
        super().__init__()
        self.cached_train = train.shuffle(shuffle).batch(train_batch).cache()
        self.cached_test = test.batch(test_batch).cache()

        # str_features = ['publication_id', 'nombre', 'categoria', 'user_id']
        # text_features = ['descripcion']
        # # int_features = ["user_gender", "bucketized_user_age"]

        # self._all_features = str_features + text_features
        # self._embeddings = {}

        self.query_model = QueryModel(
            layer_sizes=deep_layer_sizes_towers, 
            embedding_dimension=embedding_dimension
        
        )
        self.candidate_model = CandidateModel(
            layer_sizes=deep_layer_sizes_towers, 
            embedding_dimension=embedding_dimension
        )

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes_ranking]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )


    def call(self, features: Dict) -> tf.Tensor:
        query_embeddings = self.query_model(features)
        candidate_embeddings = self.candidate_model(features)

        x = tf.concat([query_embeddings, candidate_embeddings], axis=1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)


    def compute_loss(self, features: Dict, training: bool=False):
        labels = features.pop("rating")
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )
    
    
    def fit_model(self, learning_rate: float = 0.1, num_epochs: int = 1) -> None:
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate))
        model.fit(self.cached_train, epochs=num_epochs)

    
    def evaluate_model(self) -> None:
        model = self
        metrics = model.evaluate(self.cached_test, return_dict=True)
        print(metrics)


    def save_model(self, path: str) -> None:
        tf.saved_model.save(self, path)