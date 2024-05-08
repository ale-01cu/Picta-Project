from .DataPipelineBase import DataPipelineBase
import typing as typ
import pandas as pd

class DataPipelineLikes(DataPipelineBase):
    def __call__(self,
        df_to_merge: pd.DataFrame,       
    ) -> typ.Any:
        features = [
            'user_id', 
            'id', 
            'nombre', 
            'like_dislike', 
            'timestamp'
        ]
        
        df = self.merge_data(
            df_to_merge=df_to_merge,
            left_on='publication_id',
            right_on='id',
            output_features=features
        )

        df['nombre'] = df['nombre'].astype(str)
        ds = self.convert_to_tf_dataset(df)
        vocabularies = self.build_vocabularies(
            ds=ds,
            batch=1_000,
            features=features
        )

        total, train_Length, val_length, test_length = self.get_lengths(ds)


        train, val, test = self.split_into_train_and_test(
            ds=ds,
            shuffle=100_000,
            train_length=train_Length,
            val_length=val_length,
            test_length=test_length,
            seed=42
        )

        return train, val, test, vocabularies


    

