from typing import Any
import pandas as pd
from .BaseGenerator import BaseGenerator
import random

class LikesWithTimestamp(BaseGenerator):
    def __call__(self) -> None:
        publications_ids = self.df['id'].unique().tolist()
        self.generate_table(
            publications_ids=publications_ids, 
        )


    def generate_likes_dislikes(self, num_rows):
        return [
            random.randint(0, 1) 
            for i in range(num_rows) 
        ]


    def generate_table(self, publications_ids):
        # Genera un 50% más de filas de las que necesitas para tener suficientes después de eliminar las duplicadas
        num_rows_generated = int(self.num_rows * 1.5)
        users_ids = self.generate_users_id(num_rows_generated, self.users_ids_range)
        publications_ids = self.choice_publications_ids(publications_ids, num_rows_generated)
        ratings = self.generate_likes_dislikes(num_rows_generated)
        timestamps = self.generate_dates(num_rows_generated)

        df = pd.DataFrame({
            'user_id': users_ids, 
            'publication_id': publications_ids, 
            'like_dislike': ratings,
            'timestamp': timestamps
        })

        df = df.sort_values('timestamp', ascending=False)

        df = self.delete_duplicates(df.copy())
        # Recorta el DataFrame a la cantidad de filas que quieres
        df = df.iloc[:self.num_rows]
        df.to_csv(self.to_path, index=False)