from db.config import Base
import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship


stages = [
    "retrieval",
    "ranking",
    "re_ranking"
]

class Model(Base):
    __tablename__ = 'models'

    id = Column(UUID(
        as_uuid=True), 
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
        comment="Indica si el modelo se esta usando por algun engine."
    )

    stage = Column(
        Enum(*stages, name="stage_enum"),
        comment="Indica en que etapa de las tres que hay esta posicionado el modelo"
    )
    
    model_path = Column(
        String, 
        nullable=False, 
        comment="Direccion del modelo."
    )
    
    data_train_path = Column(
        String, 
        nullable=False, 
        comment="Direccion de los datos de entrenamiento del modelo."
    )

    metadata_path = Column(
        String,
        nullable=False,
        comment="Direccion de los metadatos del modelo."
    )

    engine = relationship(
        'Engine',
        back_populates="models"
    )
    
    engine_id = Column(
        UUID(as_uuid=True), 
        ForeignKey('engines.id'),
        comment="Engine al que pertenece."
    )
    
    createAt = Column(
        DateTime, 
        default=datetime.datetime.utcnow,
        comment="Fecha de creado"
    )

    def __repr__(self):
        return f"Engine(nombre={self.name}, fecha_creacion={self.createAt})"

