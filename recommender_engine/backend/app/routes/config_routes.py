from app.schemas.ModelConfigSchema import ModelConfigUserInput, ModelConfig
from fastapi import APIRouter, HTTPException, status, Body, Response
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from settings.mongodb import db
from engine.data.FeaturesTypes import features_types_map
from bson import ObjectId

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/config")
async def list_config():
    collection = db['ModelConfigCollection']
    try:
        # Filtrar por configuraciones activas y convertir _id a string
        data = [
            {**documento, '_id': str(documento['_id'])}
            for documento in collection.find({"is_active": True})
        ]
        return templates.TemplateResponse(
            "config_list.html", 
            {"request": {}, "configs": data}
        )
    except Exception as e:
        # Manejar errores de base de datos
        print(f"Error al listar configuraciones: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener configuraciones"
        )

@router.get("/config/{config_id}")
async def get_config(config_id: str):
    collection = db['ModelConfigCollection']
    try:
        # Convertir el config_id a ObjectId
        documento = collection.find_one({
            "_id": ObjectId(config_id), 
            "is_active": True
        })
        if documento:
            documento['_id'] = str(documento['_id'])
            return templates.TemplateResponse(
                "config_detail.html", 
                {"request": {}, "config": documento}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuración no encontrada"
            )
    except Exception as e:
        print(e)
        # Manejar el caso en que el config_id no sea un ObjectId válido
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de configuración no válido"
        )


@router.get("/add-config", response_class=HTMLResponse)
async def new_config_form():
    form_fields = ModelConfigUserInput.model_fields.keys()
    # Renderizar un formulario para crear una nueva configuración
    return templates.TemplateResponse(
        "new_config_form.html",
        {"request": {}, "form_fields": form_fields}
    )


def get_empty_value(type_hint):
    if type_hint == bool:
        return False
    elif type_hint == str:
        return ""
    elif type_hint == int:
        return 0
    elif type_hint == float:
        return 0.0
    elif type_hint == list:
        return []
    elif type_hint == dict:
        return {}
    else:
        return None


from typing import get_type_hints
@router.post("/add-config")
async def add_config(config: ModelConfigUserInput):
    collection = db['ModelConfigCollection']
    try:
        full_config = {
            field: get_empty_value(type_hint) 
            for field, type_hint in get_type_hints(ModelConfig).items()
        }
        config = config.model_dump()
        config = {**full_config, **config}
        # Inserta el documento y devuelve el ID insertado
        result = collection.insert_one(config)
        return { "id": str(result.inserted_id) }  # Asegúrate de convertir el ObjectId a string
    except Exception as e:
        print(f"Error al insertar configuración: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al insertar configuración"
        )
    
@router.get("/update-config/{config_id}", response_class=HTMLResponse)
async def update_config_form(config_id: str):
    collection = db['ModelConfigCollection']
    # try:
        # Convertir el config_id a ObjectId
    documento = collection.find_one({"_id": ObjectId(config_id)})
    if documento:
        fearues_types = {
            key: value().label 
            for key, value in features_types_map.items() 
        }

        documento['_id'] = str(documento['_id'])
        return templates.TemplateResponse(
            "update_config_form.html",
            {"request": {}, 'config': documento, "features_types": fearues_types}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Configuración no encontrada"
        )
    # except Exception as e:
    #     print(e)
    #     # Manejar el caso en que el config_id no sea un ObjectId válido
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="ID de configuración no válido"
    #     )

@router.put("/update-config/{config_id}")
async def update_config(config_id: str, config: ModelConfigUserInput):
    collection = db['ModelConfigCollection']
    try:
        # Actualizar el documento con los nuevos datos
        result = collection.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": config.model_dump()}
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuración no encontrada"
            )
        return {"updated_id": config_id}
    except Exception as e:
        print(f"Error al actualizar configuración: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de configuración no válido o error al actualizar"
        )


@router.get("/delete-config/{config_id}")
async def deactivate_config(config_id: str):
    collection = db['ModelConfigCollection']
    try:
        result = collection.update_one(
            {"_id": ObjectId(config_id)},
            {"$set": {"is_active": False}}
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuración no encontrada"
            )
        # Cambiar el redireccionamiento para que sea una respuesta HTML
        return await list_config()
    except Exception as e:
        # Manejar el caso en que el config_id no sea un ObjectId válido
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID de configuración no válido"
        )