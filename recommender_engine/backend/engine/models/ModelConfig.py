import json
import copy
from engine.data.FeaturesTypes import features_types_map

class ModelConfig:
    def __init__(self, 
        model_name, 
        features, 
        candidate_data_path, 
        data_path, 
        user_id_data, 
        isTrain=False, 
        **kwargs
    ):
        self.model_name = model_name
        self.features = features
        self.candidate_data_path = candidate_data_path
        self.data_path = data_path
        self.isTrain = isTrain
        self.candidate_feature_merge = kwargs.get('candidate_feature_merge', '')
        self.data_feature_merge = kwargs.get('data_feature_merge', '')
        self.towers_layers_sizes = kwargs.get('towers_layers_sizes', [])
        self.deep_layers_sizes = kwargs.get('deep_layers_sizes', [])
        self.shuffle = kwargs.get('shuffle', 10_000)
        self.embedding_dimension = kwargs.get('embedding_dimension', 64)
        self.candidates_batch = kwargs.get('candidates_batch', 128)
        self.k_candidates = kwargs.get('k_candidates', 100)
        self.learning_rate = kwargs.get('learning_rate', 0.1)
        self.num_epochs = kwargs.get('num_epochs', 1)
        self.use_multiprocessing = kwargs.get('use_multiprocessing', True)
        self.workers = kwargs.get('workers', 4)
        self.train_batch = kwargs.get('train_batch', 8192)
        self.val_batch = kwargs.get('val_batch', 4096)
        self.test_batch = kwargs.get('test_batch', 4096)
        self.vocabularies_batch = kwargs.get('vocabularies_batch', 1000)
        self.train_length = kwargs.get('train_length', 60)
        self.test_length = kwargs.get('test_length', 20)
        self.val_length = kwargs.get('val_length', 20)
        self.seed = kwargs.get('seed', 42)
        self.user_id_data = user_id_data
        self.features_data_q = kwargs.get('features_data_q', {})
        self.features_data_c = kwargs.get('features_data_c', {})
        self.target_column = kwargs.get('target_column', {})
        self.to_map = kwargs.get("to_map", False)
        
        self.learning_rate_tunning = kwargs.get('learning_rate_tunning', 0.01)
        self.train_batch_tunning = kwargs.get('train_batch_tunning', 8)
        self.val_batch_tunning = kwargs.get('val_batch_tunning', 4)
        self.test_batch_tunning = kwargs.get('test_batch_tunning', 4)

        for key, value in self.user_id_data.items():
            self.features_data_q[key] = value

        if self.to_map: 
            new_dict = self.replace_name_with_class(self.__dict__)
            self.features_data_q = new_dict['features_data_q']
            self.features_data_c = new_dict['features_data_c']
            # self.user_id_data = new_dict['user_id_data']


    def __str__(self):
        attributes = vars(self)
        # Calcular el ancho máximo para las columnas
        max_key_length = max(len(str(key)) for key in attributes.keys())
        max_value_length = max(len(str(value)) for value in attributes.values())

        # Crear la cabecera de la tabla
        header = f"{'Atributo'.ljust(max_key_length)} | {'Valor'.ljust(max_value_length)}"
        separator = '-' * (max_key_length + max_value_length + 3)

        # Crear las filas de la tabla
        rows = [f"{str(key).ljust(max_key_length)} | {str(value).ljust(max_value_length)}" for key, value in attributes.items()]

        # Unir todo en una sola cadena
        return f"{header}\n{separator}\n" + "\n".join(rows)
    
    def replace_name_with_class(self, d):
        data = copy.deepcopy(d)
        features = ['features_data_q', 'features_data_c']

        for feature in features:
            for key, value in data[feature].items():
                print(value)
                data[feature][key]['dtype'] = features_types_map[value['dtype']]
                data[feature][key]['w'] = int(value['w'])

        return data
    
    def replace_class_with_name(self, d):
        data = copy.deepcopy(d)
        features = ['features_data_q', 'features_data_c']

        for feature in features:
            for key, value in data[feature].items():
                data[feature][key]['dtype'] = value['dtype'].__name__

        return data
                
    
    def save_as_json(self, path):
        config: dict = self.replace_class_with_name(self.__dict__)

        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    # def get_config(self):
    #     return {
    #         'model_name': self.model_name,
    #         'features': self.features,
    #         'candidate_data_path': self.candidate_data_path,
    #         'data_path': self.data_path,
    #         'user_id_data': self.user_id_data,
    #         'isTrain': self.isTrain,
    #         'candidate_feature_merge': self.candidate_feature_merge,
    #         'data_feature_merge': self.data_feature_merge,
    #         'towers_layers_sizes': self.towers_layers_sizes,
    #         'deep_layers_sizes': self.deep_layers_sizes,
    #         'shuffle': self.shuffle,
    #         'embedding_dimension': self.embedding_dimension,
    #         'candidates_batch': self.candidates_batch,
    #         'k_candidates': self.k_candidates,
    #         'learning_rate': self.learning_rate,
    #         'num_epochs': self.num_epochs,
    #         'use_multiprocessing': self.use_multiprocessing,
    #         'workers': self.workers,
    #         'train_batch': self.train_batch,
    #         'val_batch': self.val_batch,
    #         'test_batch': self.test_batch,
    #         'vocabularies_batch': self.vocabularies_batch,
    #         'train_length': self.train_length,
    #         'test_length': self.test_length,
    #         'val_length': self.val_length,
    #         'seed': self.seed,
    #         'features_data_q': self.features_data_q,
    #         'features_data_c': self.features_data_c,
    #         'target_column': self.target_column,
    #         'to_map': self.to_map,
    #     }
    