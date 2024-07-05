# tasks.py
import pandas as pd
from apps.pubs.models import ClicksModel

def read_dataset():
    print("Leyendo los clicks...")
    df = pd.read_csv(ClicksModel().db_filename)
    # do something with the dataframe
    return df