import pandas as pd
import os

dirname = os.path.dirname(__file__)
likes_path = os.path.join(dirname, "../datasets/likes.csv")

df = pd.read_csv(likes_path)

comparative = df['valor'].value_counts()

print(comparative)