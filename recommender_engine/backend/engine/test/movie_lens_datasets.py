import tensorflow_datasets as tfds
import pprint

def get_datasets():
  ratings = tfds.load("movielens/100k-ratings", split="train")
  # Features of all the available movies.
  movies = tfds.load("movielens/100k-movies", split="train")

  for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)

  ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
    "movie_title": x["movie_title"],
    "bucketized_user_age": int(x["bucketized_user_age"]),
  })
  movies = movies.map(lambda x: {
    "movie_id": x["movie_id"],
    "movie_title": x["movie_title"],
  
  })

  for x in ratings.take(5).as_numpy_iterator():
    pprint.pprint(x)

  for x in movies.take(5).as_numpy_iterator():
    pprint.pprint(x)

  return ratings, movies   
            
if __name__ == "__main__":
  print(get_datasets())