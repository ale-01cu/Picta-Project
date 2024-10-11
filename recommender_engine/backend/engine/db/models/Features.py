from settings.db import Base
from sqlalchemy import Column, String
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

class Features(Base):
    __tablename__ = "features"

    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )

    name = Column(String)

    modelConfig_features = relationship(
        'ModelConfigFeatures',
        back_populates="features"
    )