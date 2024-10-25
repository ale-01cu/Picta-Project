from pydantic import BaseModel
from typing import Optional

class ModelConfigUserInput(BaseModel):
    is_active: bool
    isTrain: bool
    name: str
    stage: str
    features: list
    candidate_data_path: str
    data_path: str
    k_candidates: int
    candidate_feature_merge: str
    data_feature_merge: str
    user_id_data: dict
    features_data_q: dict
    features_data_c: dict
    target_column: Optional[dict] = None

    

class ModelConfig(BaseModel):
    is_active: bool
    isTrain: bool
    name: str
    stage: str
    features: list
    candidate_data_path: str
    data_path: str
    towers_layers_sizes: list
    shuffle: int
    embedding_dimension: int
    candidates_batch: int
    k_candidates: int
    learning_rate: float
    num_epochs: int
    use_multiprocessing: bool
    workers: int
    train_batch: int
    val_batch: int
    test_batch: int
    vocabularies_batch: int
    train_length: int
    test_length: int
    val_length: int
    seed: int
    candidate_feature_merge: str
    data_feature_merge: str
    user_id_data: dict
    features_data_q: dict
    features_data_c: dict
    deep_layers_sizes: list
    target_column: Optional[dict] = None
    to_map: bool
    stage: str
    modelPath: str
    data_train_path: str
    metadata_path: str