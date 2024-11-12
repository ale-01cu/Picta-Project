from fastapi import APIRouter, HTTPException, status, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from settings.mongodb import token_collection

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/tokens_list", response_class=HTMLResponse)
async def get_tokens_list(request: Request):
    try:
        tokens = token_collection.find()

        return templates.TemplateResponse(
            "./authorizate_plataforms/tokens_list.html", 
            {"request": request, "tokens": tokens}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )
    
@router.get("/generate_token", response_class=HTMLResponse)
async def get_generate_token(request: Request):
    try:
        return templates.TemplateResponse(
            "./authorizate_plataforms/generate_token.html", 
            {"request": request}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@router.post("/generate_token", response_class=HTMLResponse)
async def generate_token(
    request: Request, 
    name: str = Form(...),
    secret_key: str = Form(...)    
):
    try:
        # Generar el token
        expiration = datetime.utcnow() + timedelta(days=30)
        token_data = {
            "name": name,
            "exp": expiration
        }

        token = jwt.encode(
            token_data, 
            secret_key, 
            algorithm="HS256"
        )

        # Guardar el token en la base de datos
        token_collection.update_one(
            {"name": name},
            {"$set": {"token": token}},
            upsert=True
        )

        return await get_tokens_list(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )