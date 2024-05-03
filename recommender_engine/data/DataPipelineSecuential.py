from typing import Tuple, Dict, Text, List
from .DataPipelineBase import DataPipelineBase
import pandas as pd
import tensorflow as tf
from ..utils.secuential_list import group

class DataPipelineSecuential(DataPipelineBase):
    def __call__(self,
        df_to_merge: pd.DataFrame,
        # features: Dict[Text, List[str]]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        
        # data = group(self.dataframe)

        df = self.merge_data(
            df_to_merge=df_to_merge,
            left_on='publication_id',
            right_on='id',
            output_features=['user_id', 'publication_id', 'nombre']
        )

        ds = self.convert_to_tf_dataset(df)

        vocabularies = self.build_vocabularies(
            features=['publication_id', 'nombre'], 
            ds=ds, 
            batch=1_000
        )

        _, train_Length, test_length = self.get_lengths(ds)

        train, test = self.split_into_train_and_test(
            ds=ds, 
            shuffle=100_000, 
            train_length=train_Length,
            test_length=test_length,
            seed=42
        )

        group(train, 
            features=['publication_id', 'nombre'],
            q_features=['context_id', 'context_nombres'],
            c_features=['label_id', 'label_nombre']
            
        )

        return train, test, vocabularies
