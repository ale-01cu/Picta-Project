import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Dict, Text, Tuple
import math

class DataPipelineBase():
    # El dataframe con el que se va a entrenar el modelo
    dataframe: pd.DataFrame
    dataframe_path: str
    use_processed: bool

    def __init__(self, dataframe_path: str) -> None:
        self.dataframe_path = dataframe_path
        self.dataframe = pd.read_csv(self.dataframe_path)
        self.total = 0
        self.train_Length = 0 
        self.test_length = 0
        self.merge = []


    def __str__(self) -> Text:
        return f"DataPipeline(data_path={self.dataframe_path}, total={self.total}, train_Length={self.train_Length}, test_length={self.test_length}, merged={self.merge})"


    def merge_data(self, 
        df_to_merge: pd.DataFrame, 
        left_on: str,
        right_on: str, 
        output_features: list[str]
    ) -> pd.DataFrame:
        
        new_df = self.dataframe.merge(
            df_to_merge, 
            how='inner', 
            left_on=left_on, 
            right_on=right_on
        )[output_features]

        self.merge.append(df_to_merge.columns)
         
        return new_df

    
    def convert_to_tf_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices(dict(df))
    

    def build_vocabularies(self, 
        features: list[str], 
        ds: tf.data.Dataset,
        batch: int
    ) -> Dict[Text, tf.Tensor]:
        
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
    

    def get_lengths(self, ds: tf.data.Dataset) -> None:
        total = len(ds)
        train_Length = math.ceil(total * (70 / 100))
        test_length = int(total * (30 / 100))
        val_length = int(test_length * (20 / 100))
        test_length = int(test_length * (10 / 100))

        return total, train_Length, val_length, test_length


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
        tf.data.experimental.save(
            dataset, 
            path
        )


    def load(self, path:str) -> tf.data.Dataset:
        return tf.data.experimental.load(
            path, 
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
