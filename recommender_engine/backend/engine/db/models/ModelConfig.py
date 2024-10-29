from settings.db import Base
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Float
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    model_name = Column(String)
    features = relationship(
        'ModelConfigFeatures',
        back_populates="model_configs"
    )
    data_paths = Column(String)

    towers_layers_sizes = Column(String)
    deep_layers_sizes = Column(String)
    shuffle = Column(Integer, default=10_000)
    embedding_dimension = Column(Integer, default=64)
    candidates_batch = Column(Integer, default=128)
    k_candidates = Column(Integer, default=100)
    learning_rate = Column(Float, default=0.1)
    num_epochs = Column(Integer, default=1)
    use_multiprocessing = Column(Boolean, default=True)
    workers = Column(Integer, default=4)
    train_batch = Column(Integer, default=8192)
    val_batch = Column(Integer, default=4096)
    test_batch = Column(Integer, default=4096)
    vocabularies_batch = Column(Integer, default=1000)
    train_length = Column(Integer, default=60)
    test_length = Column(Integer, default=20)
    val_length = Column(Integer, default=20)
    seed = Column(Integer, default=42)

    features_data_q = Column(String)
    features_data_c = Column(String)
    target_column = Column(String)
    to_map = Column(Boolean, default=False)
