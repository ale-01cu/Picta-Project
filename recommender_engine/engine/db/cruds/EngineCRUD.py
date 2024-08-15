from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from db.models.Engine import Engine

class EngineCRUD:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()

    def create(self, name):
        try:
            engine = Engine(
                name=name
            )
            self.session.add(engine)
            self.session.commit()
            return engine
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"Error creating engine: {e}")
        # finally:
        #     session.close()

    def readAll(self):
        try:
            return self.session.query(Engine).all()
        except Exception as e:
            raise e

    def read(self, id=None, name=None):
        try:
            if id:
                engine = self.session.query(Engine).filter_by(id=id).first()
            elif name:
                engine = self.session.query(Engine).filter_by(name=name).all()
            else:
                engines = self.session.query(Engine).all()
                return engines
            return engine

        except Exception as e:
            raise e

    def update(self, id, name=None, status=None):
        try:
            engine = self.session.query(Engine).filter_by(id=id).first()
            if engine:
                if name:
                    engine.name = name
                if status != None:
                    engine.status = status
                self.session.commit()
                return engine
            else:
                raise ValueError(f"Engine with id {id} not found")

        except Exception as e:
            raise e

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

    def turn_off_all(self):
        try:
            engines = self.read()
            if engines:
                for engine in engines:
                    if engine.status:
                        self.update(engine.id, status=False)
        
        except Exception as e:
            raise e

    def turn_off_engine(self, name):
        try:
            engines = self.session.query(Engine).filter_by(status=True).all()
            if engines:
                for engine in engines:
                    self.update(engine.id, status=False)
        
        except Exception as e:
            raise e
        
        
    def turn_off_engines(self, exceptions_ids: list[str]):
        try:
            engines = self.session.query(Engine).filter_by(status=True).all()
            for engine in engines:
                print(engine.id in exceptions_ids)
                if engine.id not in exceptions_ids:
                    print(engine.name, engine.status)
                    engine.status = False
        
        except Exception as e:
            raise e


        # finally:
        #     self..close()