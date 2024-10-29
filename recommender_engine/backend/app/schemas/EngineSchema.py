from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class EngineUserInput(BaseModel):
    name: str
    is_active: bool
    is_training_by: bool
    retrieval_model_id: Optional[str]
    ranking_model_id: Optional[str]

class EngineSchema(BaseModel):
    name: str
    is_active: bool
    is_training_by: bool
    service_models_path: str
    createAt: datetime
    retrieval_model_id: Optional[int]
    ranking_model_id: Optional[int]