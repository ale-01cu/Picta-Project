import tensorflow_recommenders as tfrs
import tensorflow as tf
from .data_pipeline import vocabularies, train, test
from .QueryModel import QueryModel
from .CandidateModel import CandidateModel

class DCN(tfrs.Model):

    def __init__(self, use_cross_layer, deep_layer_sizes, 
        projection_dim=None, layer_sizes=None):
        super().__init__()

        self.embedding_dimension = 32

        str_features = ['publication_id', 'nombre', 'categoria', 'user_id']
        text_features = ['descripcion']
        # int_features = ["user_gender", "bucketized_user_age"]

        self._all_features = str_features + text_features
        self._embeddings = {}

        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim,
                kernel_initializer="glorot_uniform")
        else:
            self._cross_layer = None

        self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
        )

        self.cached_train = train.shuffle(100_000).batch(8192).cache()
        self.cached_test = test.batch(4096).cache()


    def call(self, features):
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


    def compute_loss(self, features, training=False):
        labels = features.pop("rating")
        scores = self(features)
        return self.task(
            labels=labels,
            predictions=scores,
        )
    
    
    def fit_model(self, learning_rate=0.1, num_epoch=1):
        model = self
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate))
        model.fit(self.cached_train, epochs=num_epoch)

    
    def evaluate_model(self):
        model = self
        metrics = model.evaluate(self.cached_test, return_dict=True)
        print(metrics)