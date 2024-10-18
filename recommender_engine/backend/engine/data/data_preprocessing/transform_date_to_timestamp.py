import pandas as pd

def transform_date_to_timestamp(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    """
    Transforma una columna de fecha en formato string a timestamp

    Parameters:
    df (pandas.DataFrame): DataFrame que contiene la columna de fecha
    columna_fecha (str): Nombre de la columna de fecha

    Returns:
    pandas.DataFrame: DataFrame con la columna de fecha transformada a timestamp
    """

    df[columna_fecha] = pd.to_datetime(df[columna_fecha], utc=True, errors='coerce')
    df[columna_fecha] = df[columna_fecha].astype('int64')  // 10**9
    df[columna_fecha] = df[columna_fecha].astype(int)
    return df

if __name__ == "__main__":
    transform_date_to_timestamp()