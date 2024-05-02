import pandas as pd
import random
from typing import Optional
import time
from datetime import datetime


class BaseGenerator():
    num_rows: int
    users_ids_range: int
    from_path: str
    to_path: str
    df: pd.DataFrame

    def __init__(self, 
        from_path: str, 
        to_path: str, 
        num_rows: int, 
        users_ids_range: Optional[int] = None
    ) -> None:
        
        self.num_rows = num_rows
        self.users_ids_range = users_ids_range
        self.from_path = from_path
        self.to_path = to_path
        self.df = pd.read_csv(self.from_path)


    def generate_users_id(self, num_users, users_ids_range):
        return [
            random.randint(1, users_ids_range) 
            for _ in range(num_users)
        ]
    

    def choice_publications_ids(self, id_list, amount):
        return random.choices(id_list, k=amount)


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
    
        return [
            random.randint(start, end)
            for _ in range(num_rows)
        ]
    
    def generate_time_seen(self, num_rows):
        return [
            round(random.uniform(1, 60), 2)
            for t in range(num_rows) 
        ]