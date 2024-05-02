from typing import Optional, Tuple, Dict, Text, List
import collections
import tensorflow as tf
import numpy as np


def _create_feature_dict(features: List[Text]) -> Dict[Text, List[tf.Tensor]]:
  """Crea un diccionario para las caracteristicas."""
  return {feature: [] for feature in features}

# Delvuelve una lista de tensores, cada tensor es una lista cada caracteristica
def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """
    Devuelve una lista con la cantidad de muestras de cada caracteristica.
    
    ejemplo: 

      nombre: los n nombres de ejemplo por listas
      descripcion: las n descripciones de ejemplo por listas
      ...
  """
  if random_state is None:
    random_state = np.random.RandomState()

  sampled_indices = random_state.choice(
    range(len(feature_lists["nombre"])),
    size=num_examples_per_list,
    replace=False,
  )

  sampled_features = {}

  for feature in feature_lists.keys():
    sampled_features[feature] = [
        feature_lists[feature][idx] for idx in sampled_indices
    ]

  return { 
    feature: tf.stack(feature_value, 0) 
    for feature, feature_value in sampled_features.items() 
  }


def sample_listwise(
    rating_dataset: tf.data.Dataset,
    features_for_examples: list[int],
    features_for_list: list[int],
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
  
  """ 
    Devuelve una lista de listas por cada usuario 
    construye listas de n elementos por usuario aleatoriamente
    
  """


  random_state = np.random.RandomState(seed)

  example_lists_by_user = collections.defaultdict(
    lambda: _create_feature_dict(features_for_examples))

  movie_title_vocab = set()
  for example in rating_dataset:
    user_id = example["user_id"].numpy()

    for feature in features_for_examples:
      example_lists_by_user[user_id][feature].append(
          example[feature])
    
  
      movie_title_vocab.add(example[feature].numpy())

  all_features = features_for_examples + features_for_list
  tensor_slices = {feature: [] for feature in all_features}

  for user_id, feature_lists in example_lists_by_user.items():
    for _ in range(num_list_per_user):

      # Drop the user if they don't have enough ratings.
      if len(feature_lists["nombre"]) < num_examples_per_list:
        continue

      sampled_features = _sample_list(
          feature_lists,
          num_examples_per_list,
          random_state=random_state,
      )
      tensor_slices["user_id"].append(user_id)

      for feature, feature_value in sampled_features.items():
        tensor_slices[feature].append(feature_value)

  return tf.data.Dataset.from_tensor_slices(tensor_slices)