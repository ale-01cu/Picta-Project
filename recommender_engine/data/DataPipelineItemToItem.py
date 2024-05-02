from .DataPipelineBase import DataPipelineBase
import pandas as pd

class DataPipelineItemToItem(DataPipelineBase):
    def __call__(self, df_to_merge: pd.DataFrame) -> None:
        df = self.merge_data(
            df_to_merge=df_to_merge, 
            left_on='publication_id_q',
            right_on='id',
            output_features=['publication_id_q','nombre']
        )

        df1 = self.merge_data(
            df_to_merge=df_to_merge, 
            left_on='publication_id_c',
            right_on='id',
            output_features=['publication_id_c','nombre']
        )

        df['nombre_q'] = df['nombre'].astype(str)
        df1['nombre_c'] = df1['nombre'].astype(str)


        df = pd.merge(df, df1)
        df = df.drop('nombre', axis=1)

        print(df.columns)    
        ds = self.convert_to_tf_dataset(df)

        vocabularies = self.build_vocabularies(
            features=['publication_id_q', 'nombre_q','publication_id_c','nombre_c'], 
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