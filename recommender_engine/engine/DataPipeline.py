import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Dict, Text, Tuple, List
import math
import os
import pickle

class DataPipeline:
    # El dataframe con el que se va a entrenar el modelo
    history: List[str]

    def __init__(self) -> None:
        print("*** Tuberia de datos Inicialzada ***")
        # self.total = 0
        # self.train_Length = 0 
        # self.test_length = 0
        self.history = []
        self.dirname = os.path.dirname(__file__)


    def __str__(self) -> Text:
        return self.history
    
    def get_path(self, path: str) -> str:
        return os.path.join(self.dirname, path)
    

    def read_csv_data(self, paths: List[str]) -> Tuple[pd.DataFrame]:
        print("Leyendo datos...")
        dataframes = []
        for path in paths:
            df = pd.read_csv(self.get_path(path))
            dataframes.append(df)

        return tuple(dataframes)
    
    
    def load_dataset(self, path: str):
        return tf.data.Dataset.load(path)


    def merge_data(self, 
        left_data: pd.DataFrame,
        right_data: pd.DataFrame, 
        left_on: str,
        right_on: str, 
        output_features: list[str]
    ) -> pd.DataFrame:
        print("Megeando data...")
        
        new_df = left_data.merge(
            right_data, 
            how='inner', 
            left_on=left_on, 
            right_on=right_on
        )[output_features]

        self.history.append("***** Merge *****")
        self.history.append(f"Merge de {left_data.columns} con {right_data.columns}")
        self.history.append(f"Entre {left_on} y {right_on}")
        self.history.append(f"Se obtuvo como salida: {output_features}")
        self.history.append("\n")

        return new_df

    
    def convert_to_tf_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        print("Convirtiendo data...")
        return tf.data.Dataset.from_tensor_slices(dict(data))
    

    def load_vocabularies(self, path: str):
        with open(os.path.join(path, "vocabularies.pkl"), 'rb') as f:
            vocabularies = pickle.load(f)
            return vocabularies
        

    def build_vocabularies(self, 
        features: list[str], 
        ds: tf.data.Dataset,
        batch: int
    ) -> Dict[Text, tf.Tensor]:
        
        print("Construyendo Vocabulario...")
        vocaburaries = {}
        for feature_name in features:
            vocab = ds.map(lambda x: x[feature_name], 
                num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
            dtype = ds.element_spec[feature_name].dtype

            vocaburaries[feature_name] = {}
            vocaburaries[feature_name]['dtype'] = dtype
            vocaburaries[feature_name]['vocabulary'] = np.unique(
                np.concatenate(list(vocab)))

        return vocaburaries


    def get_lengths(self, 
        ds: tf.data.Dataset,
        train_length: int,
        test_length: int,
        val_length: int    
    ) -> Tuple[int]:
        total = len(ds)
        train_length = math.ceil(total * (train_length / 100))
        val_test_length = total - train_length
        val_length = math.ceil(val_test_length * (val_length / (test_length + val_length)))
        test_length = val_test_length - val_length

        return total, train_length, val_length, test_length


    def split_into_train_and_test(
            self,
            ds: tf.data.Dataset, 
            shuffle: int, 
            train_length: int, 
            test_length: int,
            val_length: int,
            seed: int
    ) -> Tuple[tf.data.Dataset]:

        tf.random.set_seed(seed)
        shuffled = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=False)
        train = shuffled.take(train_length)
        val = shuffled.skip(train_length).take(val_length)
        test = shuffled.skip(train_length).skip(val_length).take(test_length)

        return train, val, test


    def save(self, dataset: tf.data.Dataset, path:str) -> None:
        dataset.save(
            os.path.join(self.dirname, path)
        ), 


    def load(self, path:str) -> tf.data.Dataset:
        return tf.data.Dataset.load(
            path
            # tf.TensorSpec(shape=(), dtype=tf.int64)
        )


    def data_caching(self, 
        train: tf.data.Dataset, 
        val: tf.data.Dataset,
        test: tf.data.Dataset, 
        shuffle: int,
        train_batch: int,
        val_batch: int,
        test_batch: int
    ) -> Tuple[tf.data.Dataset]: 
        cached_train = train.shuffle(shuffle)\
            .batch(train_batch).cache()
        cached_val = val.batch(val_batch).cache()
        cached_test = test.batch(test_batch).cache()

        return cached_train, cached_val, cached_test


    def close(self):
        print("*** Tuberia de datos Cerrada ***")

    # def data_pipeline_process(): 
    #     pass