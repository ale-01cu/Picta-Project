from engine.db.models.Engine import Engine
from engine.db.models.Model import Model
from settings.db import engine

def build_db():
    Model.metadata.create_all(engine)
    Engine.metadata.create_all(engine)
