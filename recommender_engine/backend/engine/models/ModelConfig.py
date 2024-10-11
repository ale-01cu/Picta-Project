import json
import copy
from engine.FeaturesTypes import features_types_map

class ModelConfig:
    def __init__(self, model_name, features, candidate_data_path, data_path, user_id_data,**kwargs):
        self.model_name = model_name
        self.features = features
        self.candidate_data_path = candidate_data_path
        self.data_path = data_path
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
        
        for key, value in self.user_id_data.items():
            self.features_data_q[key] = value

        if self.to_map: 
            new_dict = self.replace_name_with_class(self.__dict__)
            self.features_data_q = new_dict['features_data_q']
            self.features_data_c = new_dict['features_data_c']
            self.user_id_data = new_dict['user_id_data']


    def __str__(self):
        return f"ModelConfig: {self.model_name}"
    
    def replace_name_with_class(self, d):
        data = copy.deepcopy(d)
        features = ['features_data_q', 'features_data_c', 'user_id_data']

        for feature in features:
            for key, value in data[feature].items():
                data[feature][key]['dtype'] = features_types_map[value['dtype']]

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
