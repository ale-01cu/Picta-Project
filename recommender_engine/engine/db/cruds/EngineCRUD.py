from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from db.models.Engine import Engine

class EngineCRUD:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    def create(self, name):
        session = self.Session()
        try:
            engine = Engine(
                name=name
            )
            session.add(engine)
            session.commit()
            return engine
        except IntegrityError as e:
            session.rollback()
            raise ValueError(f"Error creating engine: {e}")
        finally:
            session.close()

    def read(self, id=None, name=None):
        session = self.Session()
        try:
            if id:
                engine = session.query(Engine).filter_by(id=id).first()
            elif name:
                engine = session.query(Engine).filter_by(name=name).first()
            else:
                engines = session.query(Engine).all()
                return engines
            return engine
        finally:
            session.close()

    def update(self, id, name=None):
        session = self.Session()
        try:
            engine = session.query(Engine).filter_by(id=id).first()
            if engine:
                if name:
                    engine.name = name
                session.commit()
                return engine
            else:
                raise ValueError(f"Engine with id {id} not found")
        finally:
            session.close()

    def delete(self, id):
        session = self.Session()
        try:
            engine = session.query(Engine).filter_by(id=id).first()
            if engine:
                session.delete(engine)
                session.commit()
                return True
            else:
                raise ValueError(f"Engine with id {id} not found")
        finally:
            session.close()

    def get_models(self, id):
        session = self.Session()
        try:
            engine = session.query(Engine).filter_by(id=id).first()
            if engine:
                return engine.models
            else:
                raise ValueError(f"Engine with id {id} not found")
        finally:
            session.close()