from db.models.Engine import Engine
from db.models.Model import Model
from db.config import engine

def build_db():
    Model.metadata.create_all(engine)
    Engine.metadata.create_all(engine)
