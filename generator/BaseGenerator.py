import pandas as pd
import random
from typing import Optional
import time
from datetime import datetime
import numpy as np


class BaseGenerator():
    num_rows: int
    users_ids_range: int
    from_path: str
    to_path: str
    seed: int
    df: pd.DataFrame

    def __init__(self, 
        from_path: str, 
        to_path: str, 
        num_rows: int, 
        seed: int,
        users_ids_range: Optional[int] = None
    ) -> None:
        
        self.num_rows = num_rows
        self.users_ids_range = users_ids_range
        self.from_path = from_path
        self.to_path = to_path
        self.df = pd.read_csv(self.from_path)
        self.seed = seed

        np.random.seed(self.seed)


    def generate_users_id(self, num_users, users_ids_range):
         return list(np.random.randint(1, users_ids_range, num_users))
    

    def choice_publications_ids(self, id_list, amount):
        return list(np.random.choice(id_list, amount))


    def delete_duplicates(self, df):
        df_temp = df.drop_duplicates(
            subset=['user_id', 'publication_id'], keep='first')
        df_temp = df.drop_duplicates(
            subset=['user_id', 'timestamp'], keep='first')
        return df_temp
    

    def generate_dates(
        self,
        num_rows: int,
        start: datetime = time.mktime(time.strptime('01-01-2020', '%d-%m-%Y')), 
        end: datetime = time.mktime(time.strptime('31-12-2024', '%d-%m-%Y'))
    ):
    
        return list(np.random.randint(start, end, num_rows))
    
    def generate_time_seen(self, num_rows):
        return list(np.round(np.random.uniform(1, 60, num_rows), 2))