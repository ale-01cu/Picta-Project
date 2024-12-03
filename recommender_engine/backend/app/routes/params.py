from fastapi import (
    APIRouter, 
    HTTPException, 
    status, 
)
from settings.db import engine
from bson import ObjectId
from engine.exceptions.use_engine import (
    MetadataNotFound,
)
from engine.exceptions.train import (
    ModelIdNotProvide,
    ModelNotFound
)
from settings.mongodb import engine_collection, config_collection
from engine.utils import read_json
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/params/params")
async def get_params():
    try:
        engine = engine_collection.find_one({ "is_active": True })

        if engine['retrieval_model_id']:
            retrieval_config_db = config_collection.find_one(
                { "_id": ObjectId(engine['retrieval_model_id']) })
            if not retrieval_config_db:
                raise ModelNotFound.ModelNotFoundException("Retrieval")
        else: raise ModelIdNotProvide.ModelIdNotProvideException("Retrieval")
        if engine['ranking_model_id']:
            ranking_config_db = config_collection.find_one(
                { "_id": ObjectId(engine['ranking_model_id']) })
            if not ranking_config_db:
                raise ModelNotFound.ModelNotFoundException(engine['ranking_model_id'])
        else:
            raise ModelIdNotProvide.ModelIdNotProvideException("Ranking")
        

        try: retrieval_hyperparams = read_json(retrieval_config_db['metadata_path'])
        except: raise MetadataNotFound.MetadataNotFoundException("Retrieval")
        try: ranking_hyperparams = read_json(ranking_config_db['metadata_path'])
        except: raise MetadataNotFound.MetadataNotFoundException("Ranking")

        retrieval_features_data_q = retrieval_hyperparams['features_data_q']
        ranking_feature_data_q = ranking_hyperparams['features_data_q']

        features_data = {
            **retrieval_features_data_q, 
            **ranking_feature_data_q
        }

        # Filtrar por configuraciones activas y convertir _id a string
        return features_data
    except Exception as e:
        # Manejar errores de base de datos
        print(f"Error al obtener los parametros de contexto: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener los parametros de contexto"
        )