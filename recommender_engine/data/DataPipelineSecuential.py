from typing import Tuple, Dict, Text, List
from .DataPipelineBase import DataPipelineBase
import pandas as pd
import tensorflow as tf

class DataPipelineSecuential(DataPipelineBase):
    def __call__(self,
        df_to_merge: pd.DataFrame,
        # features: Dict[Text, List[str]]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        
        # data = group(self.dataframe)

        df = self.merge_data(
            df_to_merge=df_to_merge,
            left_on='publication_id',
            right_on='id',
            output_features=['user_id', 'publication_id', 'nombre']
        )

        sampled_list = self.build_sequential_data(df, 
            features=['publication_id', 'nombre'],
            q_features=['context_id', 'context_nombres'],
            c_features=['id', 'nombre'],
            desired_size=9
            
        )


        print('Convirtiendo a Dataset...')
        ds = self.convert_to_tf_dataset(sampled_list)

        print('Creando el vocabulario...')
        vocabularies = self.build_vocabularies(
            features=['context_id', 'context_nombres', 'id', 'nombre'], 
            ds=ds, 
            batch=1_000
        )

        _, train_Length, test_length = self.get_lengths(ds)

        train, test = self.split_into_train_and_test(
            ds=ds, 
            shuffle=100_000, 
            train_length=train_Length,
            test_length=test_length,
            seed=42
        )

        return train, test, vocabularies
    

    def build_sequential_data(self,
        df: pd.DataFrame, 
        q_features: list[str], 
        c_features: list[str], 
        features: list[str],
        desired_size: int
    ) -> Dict[Text, List[tf.Tensor]]:
        
        """
            *** Parametros ***
            df: datos base de donde sacar la informacion
            q_features: las caracteristicas que debe de procesar la
            torre de consulta
            c_features: las caracteristicas que debe de procesar la
            torre candidata
            features: las caracteristicas que se quieren procesar del dataset
            pasado por parametros "df"

            importante: las tres variables de features deben de tener las mismas 
            dimensiones y las features deben de estar en el mismo orden.

            Devuelve un diccionario con las propiedades que se pasan por parametro
            y cada propiedad guarda una lista de listas, osea cada valor de cada propiedad es una lista
            pero para poder convertir eso a un dataset de tensorflow hay que guardar esas listas dentro de
            otra lista.
        
        """

        assert (len(q_features) == len(c_features) == len(features)), \
            'Las longitudes de las características no son iguales'


        sampled_fetures = { 
            feature: [] 
            for feature in q_features + c_features 
        }

        # Crear un diccionario vacío para almacenar los resultados
        for user_id, group in df.groupby('user_id'):

            for f, q, c in zip(features, q_features, c_features):
                q_values = group[f].values[:-1]
                c_values = group[f].values[-1:]

                if q_values.size == 0 or c_values.size == 0: print('usuario con cero: ', user_id)

                if q_values.size > 0:
                    sampled_fetures[q].append(tf.stack(q_values, 0))
                else:
                    # Manejar el caso en que q_values es una lista vacía
                    # Por ejemplo, podrías agregar un tensor de ceros del tamaño deseado
                    sampled_fetures[q].append(tf.stack(tf.zeros([desired_size]), 0))
                    
                if c_values.size > 0:
                    sampled_fetures[c].append(tf.stack(c_values, 0))
                else:
                    # Manejar el caso en que c_values es una lista vacía
                    # Por ejemplo, podrías agregar un tensor de ceros del tamaño deseado
                    sampled_fetures[c].append(tf.stack(tf.zeros([1]), 0))

        return sampled_fetures
    


    def evaluate_build_sequential_data(self, sampled_fetures: Dict[Text, List]) -> None:
        lists_Sizes = {}
        for i, j in sampled_fetures.items():
            lists_Sizes[i] = []
            
            if isinstance(j, list): 
                print('lista: ', i, len(j))
                for l in j:
                    size = len(l)
                    if size not in lists_Sizes[i]: 
                        lists_Sizes[i].append(size) 
            else: print('no lista: ', i, j)

        print(lists_Sizes)