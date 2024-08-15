from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from db.models.Model import Model

class ModelCRUD:
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker(bind=engine)
        self.session = self.Session()

    def create(self, 
        name, 
        stage, 
        model_path, 
        data_train_path, 
        metadata_path, 
        # engine,
        engine_id
    ):
        try:
            model = Model(
                name=name,
                stage=stage,
                model_path=model_path,
                data_train_path=data_train_path,
                metadata_path=metadata_path,
                # engine=engine,
                engine_id=engine_id
            )
            self.session.add(model)
            self.session.commit()
            return model
        except IntegrityError as e:
            self.session.rollback()
            raise ValueError(f"Error creating model: {e}")
        # finally:
        #     session.close(
        
    def readAll(self):
        try:
            return self.session.query(Model).all()
        except Exception as e:
            raise e

    def read(self, id=None, name=None):
        try:
            if id:
                model = self.session.query(Model).filter_by(id=id).first()
            elif name:
                model = self.session.query(Model).filter_by(name=name).all()
            else:
                models = self.session.query(Model).all()
                return models
            return model

        except Exception as e:
            raise e

    def update(self, id, name=None, status=None, model_path=None, data_train_path=None, metadata_path=None, engine_id=None):
        try:
            model = self.session.query(Model).filter_by(id=id).first()
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
                if status != None:
                    model.status = status
                self.session.commit()
                return model
            else:
                raise ValueError(f"Model with id {id} not found")

        except Exception as e:
            raise e

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

    def turn_off_all(self):
        try:
            models = self.read()
            if models:
                for model in models:
                    if model.status:
                        self.update(model.id, status=False)
        
        except Exception as e:
            raise e

    def turn_off_model(self, id, name, stage):
        try:
            models = self.read(name=name).filter(Model.id != id)
            if models:
                for model in models:
                    if model.stage == stage and model.status:
                        self.update(model.id, status=False)

        except Exception as e:
            raise e
        

    def turn_off_models(self, exceptions_ids: list[str]):
        try:
            models = self.session.query(Model).filter_by(status=True).all()
            for model in models:
                if model.id not in exceptions_ids:
                    model.status = False
        
        except Exception as e:
            raise e