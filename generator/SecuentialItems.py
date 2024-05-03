from typing import Any
from .BaseGenerator import BaseGenerator
import pandas as pd
import time

class SecuentialItems(BaseGenerator):
    def __call__(self, k_sequence = 11) -> None:
        publications_ids = self.df['id'].unique().tolist()
        self.generate_table(
            publications_ids=publications_ids, 
            k_sequence=k_sequence
        )


    def delete_duplicates(self, df):
        return df.drop_duplicates(subset=[
            'user_id', 'publication_id'], keep='first')


    def generate_table(self, publications_ids, k_sequence: int):
        # Genera un 50% más de filas de las que necesitas para tener suficientes después de eliminar las duplicadas
        data = {
            'user_id': [],
            'publication_id': []
        }


        for i in range(1, self.num_rows+1):
            users_ids = [i] * k_sequence
            publications_ids = self.choice_publications_ids(
                publications_ids, k_sequence)
            
            # time.sleep(1)
            # print(data)

            data['user_id'] += users_ids
            data['publication_id'] += publications_ids


        df = pd.DataFrame(data)

        # df = df.sort_values('timestamp', ascending=False)

        # df = self.delete_duplicates(df.copy())
        # Recorta el DataFrame a la cantidad de filas que quieres
        # df = df.iloc[:self.num_rows]
        df.to_csv(self.to_path, index=False)