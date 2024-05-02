from typing import Any
import random
from datetime import datetime
import time
import pandas as pd
from .BaseGenerator import BaseGenerator

class ItemToItem(BaseGenerator):
    def __call__(self) -> Any:
        publications_ids = self.df['id'].unique().tolist()
        self.generate_table(
            publications_ids=publications_ids, 
        )


    def choice_publications_ids(self, id_list, amount):
        return random.choices(id_list, k=amount)
    

    def delete_duplicates(self, df):
        df_temp = df.drop_duplicates(subset=['publication_id_q', 'publication_id_c'], keep='first')
        return df_temp
    

    def generate_dates(
        self,
        num_rows: int,
        start: datetime = time.mktime(time.strptime('01-01-2020', '%d-%m-%Y')), 
        end: datetime = time.mktime(time.strptime('31-12-2024', '%d-%m-%Y'))):
    
        return [
            random.randint(start, end)
            for _ in range(num_rows)
        ]
    

    def generate_table(self, publications_ids):
        # Genera un 50% más de filas de las que necesitas para tener suficientes después de eliminar las duplicadas
        num_rows_generated = int(self.num_rows * 1.5)

        query_publications_ids = self.choice_publications_ids(
            publications_ids, num_rows_generated)
        candidate_publications_ids = self.choice_publications_ids(
            publications_ids, num_rows_generated)

        timestamps = self.generate_dates(num_rows_generated)

        df = pd.DataFrame({
            'publication_id_q': query_publications_ids,
            'publication_id_c': candidate_publications_ids, 
            'timestamp': timestamps
        })

        df = df.sort_values('timestamp', ascending=False)

        df = self.delete_duplicates(df.copy())
        # Recorta el DataFrame a la cantidad de filas que quieres
        df = df.iloc[:self.num_rows]
        df.to_csv(self.to_path, index=False)