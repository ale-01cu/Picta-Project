from settings.db import Base
from sqlalchemy import Column, String, DateTime, Boolean
import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

class Engine(Base):
    __tablename__ = 'engines'
    id = Column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )

    name = Column(
        String(50), 
        nullable=False
    )

    status = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Indica si el engine se esta usando."
    )
    
    createAt = Column(
        DateTime, 
        default=datetime.datetime.utcnow
    )
    
    models = relationship(
        "Model", 
        back_populates="engine"
    )

    # def __repr__(self):
    #     return f"Engine(nombre={self.name}, fecha_creacion={self.createAt})"
    