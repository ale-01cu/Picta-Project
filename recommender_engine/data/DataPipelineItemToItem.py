from .DataPipelineBase import DataPipelineBase
import pandas as pd
from typing import Tuple, Dict, Text, List
import tensorflow as tf

class DataPipelineItemToItem(DataPipelineBase):
    def __call__(self,
        df_to_merge: pd.DataFrame,
        # features: Dict[Text, List[str]]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        df = self.merge_data(
            df_to_merge=df_to_merge,
            left_on='publication_id_q',
            right_on='id',
            output_features=['publication_id_q','nombre']
        )

        df['nombre_q'] = df['nombre'].astype(str)
        df = df.drop('nombre', axis=1)

        df1 = self.merge_data(
            df_to_merge=df_to_merge,
            left_on='publication_id_c',
            right_on='id',
            output_features=['publication_id_c','nombre']
        )

        df1['nombre'] = df1['nombre'].astype(str)
        df1['id'] = df1['publication_id_c']
        df1 = df1.drop('publication_id_c', axis=1)

        df = pd.concat([df, df1], axis=1)

        ds = self.convert_to_tf_dataset(df)

        vocabularies = self.build_vocabularies(
            features=['publication_id_q', 'nombre_q', 'id', 'nombre'],
            ds=ds,
            batch=1_000
        )

        total, train_Length, test_length = self.get_lengths(ds)

        train, test = self.split_into_train_and_test(
            ds=ds,
            shuffle=100_000,
            train_length=train_Length,
            test_length=test_length,
            seed=42
        )


        return train, test, vocabularies