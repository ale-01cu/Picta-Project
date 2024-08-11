from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from db.models.Model import Model

class ModelCRUD:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)

    def create(self, name, stage, model_path, data_train_path, metadata_path, engine_id):
        session = self.Session()
        try:
            model = Model(
                name=name,
                stage=stage,
                model_path=model_path,
                data_train_path=data_train_path,
                metadata_path=metadata_path,
                engine_id=engine_id
            )
            session.add(model)
            session.commit()
            return model
        except IntegrityError as e:
            session.rollback()
            raise ValueError(f"Error creating model: {e}")
        finally:
            session.close()

    def read(self, id=None, name=None):
        session = self.Session()
        try:
            if id:
                model = session.query(Model).filter_by(id=id).first()
            elif name:
                model = session.query(Model).filter_by(name=name).first()
            else:
                models = session.query(Model).all()
                return models
            return model
        finally:
            session.close()

    def update(self, id, name=None, model_path=None, data_train_path=None, metadata_path=None, engine_id=None):
        session = self.Session()
        try:
            model = session.query(Model).filter_by(id=id).first()
            if model:
                if name:
                    model.name = name
                if model_path:
                    model.model_path = model_path
                if data_train_path:
                    model.data_train_path = data_train_path
                if metadata_path:
                    model.metadata_path = metadata_path
                if engine_id:
                    model.engine_id = engine_id
                session.commit()
                return model
            else:
                raise ValueError(f"Model with id {id} not found")
        finally:
            session.close()

    def delete(self, id):
        session = self.Session()
        try:
            model = session.query(Model).filter_by(id=id).first()
            if model:
                session.delete(model)
                session.commit()
                return True
            else:
                raise ValueError(f"Model with id {id} not found")
        finally:
            session.close()

    def turn_off_model(self, name, stage, status):
        session = self.Session()

        try:
            models = self.read(name=name)
            if models:
                for model in models:
                    if model.stage == stage and model.status == status:
                        self.update(model.id, status=False)

        finally:
            session.close()
