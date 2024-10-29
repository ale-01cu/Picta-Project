import numpy as np

def add_age_column(df, column_name):
    new_df = df.copy()
    new_df[column_name] = np.random.normal(40, 10, size=len(df)).astype(int)
    return new_df

if "__main__" == __name__:
    add_age_column()