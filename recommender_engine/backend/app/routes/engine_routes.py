from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.requests import Request
from bson import ObjectId
from settings.mongodb import engine_collection, config_collection
from app.schemas.EngineSchema import EngineUserInput
from fastapi.templating import Jinja2Templates
from datetime import datetime

router = APIRouter()
templates = Jinja2Templates(directory="app/templates/engine")


# List all engines
@router.get("/engines", response_class=HTMLResponse)
async def list_engines(request: Request):
    try:
        engines = engine_collection.find().to_list(100)
        return templates.TemplateResponse(
            "list_engines.html", 
            {"request": request, "engines": engines}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Retrieve a single engine by ID
@router.get("/engines/{engine_id}", response_class=HTMLResponse)
async def retrieve_engine(request: Request, engine_id: str):
    try:
        engine = engine_collection.find_one({"_id": ObjectId(engine_id)})
        if engine is None:
            raise HTTPException(status_code=404, detail="Engine not found")
        
        # if not engine.get('retrieval_model_id') or not engine.get('ranking_model_id'):
        #     return templates.TemplateResponse(
        #         "detail_engine.html",
        #         {"request": request, "engine": engine}
        #     )
        
        if engine['retrieval_model_id']:
            engine['retrieval_model_id'] = config_collection.find_one({
                    "_id": ObjectId(engine['retrieval_model_id']), 
                    "is_active": True
                })
        if engine['ranking_model_id']:
            engine['ranking_model_id'] = config_collection.find_one({
                    "_id": ObjectId(engine['ranking_model_id']), 
                    "is_active": True
                })
        
        return templates.TemplateResponse(
            "detail_engine.html",
            {"request": request, "engine": engine}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create a new engine
@router.post("/engines")
async def create_engine(engine: EngineUserInput):
    try:
        existing_engine = engine_collection.find_one({"name": engine.name})
        if existing_engine:
            raise HTTPException(
                status_code=400, 
                detail="El nombre del engine ya existe"
            )

        if engine.is_active:
            active_engine = engine_collection.find_one({"is_active": True})
            if active_engine:
                raise HTTPException(status_code=400, detail="Ya existe un engine activo")

        engine_data = jsonable_encoder(engine)
        engine_data['createAt'] = datetime.now()
        engine_data['service_models_path'] = ''
        new_engine = engine_collection.insert_one(engine_data)
        return { "id": str(new_engine.inserted_id) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get template for creating a new engine
@router.get("/create-engines", response_class=HTMLResponse)
async def get_create_engine_template(request: Request):
    try:
        data = [
            {**documento, '_id': str(documento['_id'])}
            for documento in config_collection.find({"is_active": True})
        ]
        return templates.TemplateResponse(
            "create_engine.html",
            {"request": request, "configs": data},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update an existing engine
@router.put("/engines/{engine_id}")
async def update_engine(engine_id: str, engine: EngineUserInput):
    try:
        existing_engine = engine_collection.find_one({"name": engine.name, "_id": {"$ne": ObjectId(engine_id)}})
        if existing_engine:
            raise HTTPException(status_code=400, detail="El nombre del engine ya existe")

        if engine.is_active:
            active_engine = engine_collection.find_one({"is_active": True, "_id": {"$ne": ObjectId(engine_id)}})
            if active_engine:
                raise HTTPException(status_code=400, detail="Ya existe un engine activo")

        engine_data = jsonable_encoder(engine)
        update_result = engine_collection.update_one({"_id": ObjectId(engine_id)}, {"$set": engine_data})
        if update_result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Engine not found")
        return {"updated_id": engine_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get template for updating an engine
@router.get("/edit-engines/{engine_id}", response_class=HTMLResponse)
async def get_update_engine_template(request: Request, engine_id: str):
    try:
        engine = engine_collection.find_one({"_id": ObjectId(engine_id)})
        data = [
            {**documento, '_id': str(documento['_id'])}
            for documento in config_collection.find({"is_active": True})
        ]

        if engine is None:
            raise HTTPException(status_code=404, detail="Engine not found")
        return templates.TemplateResponse(
            "update_engine.html", 
            {"request": request, "engine": engine, "configs": data}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










# # List all engines
# @router.get("/engines", response_class=HTMLResponse)
# async def list_engines(request: Request):
#     engines = await engine_collection.find().to_list(100)
#     return templates.TemplateResponse("list_engines.html", {"request": request, "engines": engines})

# # Retrieve a single engine by ID
# @router.get("/engines/{engine_id}", response_class=HTMLResponse)
# async def retrieve_engine(request: Request, engine_id: str):
#     engine = await engine_collection.find_one({"_id": ObjectId(engine_id)})
#     if engine is None:
#         raise HTTPException(status_code=404, detail="Engine not found")
#     return templates.TemplateResponse("retrieve_engine.html", {"request": request, "engine": engine})

# # Create a new engine
# @router.post("/engines", response_class=RedirectResponse)
# async def create_engine(engine: EngineUserInput):
#     engine_data = jsonable_encoder(engine)
#     new_engine = await engine_collection.insert_one(engine_data)
#     return RedirectResponse(url=f"/engines/{new_engine.inserted_id}", status_code=status.HTTP_303_SEE_OTHER)

# # Get template for creating a new engine
# @router.get("/engines/create", response_class=HTMLResponse)
# async def get_create_engine_template(request: Request):
#     return templates.TemplateResponse("create_engine.html", {"request": request})

# # Update an existing engine
# @router.put("/engines/{engine_id}", response_class=RedirectResponse)
# async def update_engine(engine_id: str, engine: EngineUserInput):
#     engine_data = jsonable_encoder(engine)
#     update_result = await engine_collection.update_one({"_id": ObjectId(engine_id)}, {"$set": engine_data})
#     if update_result.modified_count == 0:
#         raise HTTPException(status_code=404, detail="Engine not found")
#     return RedirectResponse(url=f"/engines/{engine_id}", status_code=status.HTTP_303_SEE_OTHER)

# # Get template for updating an engine
# @router.get("/engines/{engine_id}/edit", response_class=HTMLResponse)
# async def get_update_engine_template(request: Request, engine_id: str):
#     engine = await engine_collection.find_one({"_id": ObjectId(engine_id)})
#     if engine is None:
#         raise HTTPException(status_code=404, detail="Engine not found")
#     return templates.TemplateResponse("update_engine.html", {"request": request, "engine": engine})

# # Delete an engine
# @router.delete("/engines/{engine_id}", response_class=Response)
# async def delete_engine(engine_id: str):
#     delete_result = await engine_collection.delete_one({"_id": ObjectId(engine_id)})
#     if delete_result.deleted_count == 0:
#         raise HTTPException(status_code=404, detail="Engine not found")
#     return Response(status_code=status.HTTP_204_NO_CONTENT)