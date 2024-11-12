from engine.test.movie_lens_datasets import get_datasets
import tensorflow_datasets as tfds

def get_movielens_dataset_as_csv():
  ratings, movies = get_datasets()
  
  # Convertir los datasets de TensorFlow a DataFrames de pandas
  ratings_df = tfds.as_dataframe(ratings)
  movies_df = tfds.as_dataframe(movies)
  
  # Guardar los DataFrames como archivos CSV
  ratings_df.to_csv('ratings.csv', index=False)
  movies_df.to_csv('movies.csv', index=False)

if __name__ == "__main__":
  get_movielens_dataset_as_csv()