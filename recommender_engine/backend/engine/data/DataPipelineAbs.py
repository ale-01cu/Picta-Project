from abc import ABC, abstractmethod
import tensorflow as tf
import pandas as pd
from typing import Dict, Text, Tuple, List

class AbstractDataPipeline(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> Text:
        pass

    @abstractmethod
    def get_path(self, path: str) -> str:
        pass

    @abstractmethod
    def read_csv_data(self, paths: List[str]) -> Tuple[pd.DataFrame]:
        pass

    @abstractmethod
    def load_dataset(self, path: str):
        pass

    @abstractmethod
    def merge_data(self, 
        left_data: pd.DataFrame,
        right_data: pd.DataFrame, 
        left_on: str,
        right_on: str, 
        output_features: list[str]
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def convert_to_tf_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        pass

    @abstractmethod
    def load_vocabularies(self, path: str):
        pass

    @abstractmethod
    def build_vocabularies(self, 
        features: list[str], 
        ds: tf.data.Dataset,
        batch: int
    ) -> Dict[Text, tf.Tensor]:
        pass

    @abstractmethod
    def get_lengths(self, 
        ds: tf.data.Dataset,
        train_length: int,
        test_length: int,
        val_length: int    
    ) -> Tuple[int]:
        pass

    @abstractmethod
    def split_into_train_and_test(
            self,
            ds: tf.data.Dataset, 
            shuffle: int, 
            train_length: int, 
            test_length: int,
            val_length: int,
            seed: int
    ) -> Tuple[tf.data.Dataset]:
        pass

    @abstractmethod
    def save(self, dataset: tf.data.Dataset, path:str) -> None:
        pass

    @abstractmethod
    def load(self, path:str) -> tf.data.Dataset:
        pass

    @abstractmethod
    def data_caching(self, 
        train: tf.data.Dataset, 
        val: tf.data.Dataset,
        test: tf.data.Dataset, 
        shuffle: int,
        train_batch: int,
        val_batch: int,
        test_batch: int
    ) -> Tuple[tf.data.Dataset]:
        pass

    @abstractmethod
    def close(self):
        pass

    