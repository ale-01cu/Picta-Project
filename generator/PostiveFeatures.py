from typing import Any
from .BaseGenerator import BaseGenerator
import numpy as np
import pandas as pd

class PositiveFeaturesGenerator(BaseGenerator):
    def __call__(self, categories) -> Any:
        publications_ids = self.df['id'].unique().tolist()
        self.generate_table(
            publications_ids=publications_ids,
            categories=categories
        )
    
    def choice_categories(self, categories, amount):
       return list(np.random.choice(categories, amount))
    
    def delete_duplicates(self, df):
        df_temp = df.drop_duplicates(
            subset=['user_id', 'publication_id', 'category'], keep='first')
        return df_temp

    def generate_table(self, publications_ids, categories):
        num_rows_generated = int(self.num_rows * 1.5)

        users_ids = self.generate_users_id(num_rows_generated, self.users_ids_range)
        publications_ids = self.choice_publications_ids(publications_ids, num_rows_generated)
        timestamps = self.generate_dates(num_rows_generated)
        categories = self.choice_categories(categories=categories, amount=num_rows_generated)

        df = pd.DataFrame({
            'user_id': users_ids, 
            'publication_id': publications_ids, 
            'category': categories,
            'timestamp': timestamps
        })

        df = self.delete_duplicates(df.copy())
        df = df.iloc[:self.num_rows]
        df.to_csv(self.to_path, index=False)