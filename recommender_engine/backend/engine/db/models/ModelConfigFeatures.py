from settings.db import Base
from sqlalchemy import Column, String, Integer, ForeignKey
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

class ModelConfigFeatures(Base):
    __tablename__ = 'modelConfig_features'
    model_config_id = Column(Integer, ForeignKey('model_configs.id'), primary_key=True)
    model_config = relationship(
        'ModelConfig',
        back_populates="modelConfig_features"
    )
    features_id = Column(Integer, ForeignKey('features.id'), primary_key=True)
    features = relationship(
        'Features',
        back_populates="modelConfig_features"
    )