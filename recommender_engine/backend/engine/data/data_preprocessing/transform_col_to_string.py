from engine.data.DataPipeline import DataPipeline
import pandas as pd

def transform_col_to_string():
    pipe = DataPipeline()
    pubs_df, = pipe.read_csv_data(paths=[
        '../../../../datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv'
    ])

    print(pubs_df)

    data = {
        "id": [ 1, 2, 3 ],
        "value": [1.0 , 2.0 , 3.0]
    }

    df = pd.DataFrame(data)
    df['id'] = df['id'].astype(str)
    df['value'] = df['value'].astype(str)
    print(df)
    print(df.dtypes)

    print(df['id'].values)


if __name__ == "__main__":
    transform_col_to_string()