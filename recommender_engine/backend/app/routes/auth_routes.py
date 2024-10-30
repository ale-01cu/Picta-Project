from fastapi import APIRouter, HTTPException, status, Request, Form, Response
from settings.mongodb import db, user_collection
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.schemas.UserSchema import UserSchema
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/signup", response_class=HTMLResponse)
async def get_form(request: Request):
    try:
        form_fields = UserSchema.model_fields.keys()
        return templates.TemplateResponse(
            "signup.html", 
            {"request": request, "form_fields": form_fields}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@router.post("/signin", response_class=HTMLResponse)
async def signin(
    username: str = Form(...), 
    password: str = Form(...)
):
    from main import SECRET_KEY
    try:
        collection = db["users"]
        
        # Buscar el usuario por email
        user = collection.find_one({"username": username})
        
        if not user or not pwd_context.verify(password, user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Credenciales incorrectas"
            )
        
        token_data = {
            "sub": username,
            "exp": datetime.utcnow() + timedelta(days=30)  # El token expira en 1 hora
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm="HS256")
        
        # Crear una respuesta de redirección
        response = RedirectResponse(
            "/config",
            status_code=status.HTTP_303_SEE_OTHER
        )
        
        # Establecer la cookie con el token
        response.set_cookie(
            key="auth_token", 
            value=token, 
            httponly=True
        )
        
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@router.get("/signin", response_class=HTMLResponse)
async def get_signin_form(request: Request):
    try:
        form_fields = UserSchema.model_fields.keys()
        return templates.TemplateResponse(
            "signin.html", 
            {"request": request, "form_fields": form_fields}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@router.post("/signup", response_class=HTMLResponse)
async def signup(
    username: str = Form(...), 
    password: str = Form(...)
):
    collection = user_collection
    try:
        # Verificar si el usuario ya existe
        if collection.find_one({"username": username}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El nombre de usuario ya está en uso"
            )
        
        # Hashear la contraseña
        hashed_password = pwd_context.hash(password)
        
        # Crear el nuevo usuario
        new_user = {
            "username": username,
            "password": hashed_password
        }
        
        # Insertar el nuevo usuario en la base de datos
        collection.insert_one(new_user)

        return RedirectResponse(
            "/signin",
            status_code=status.HTTP_303_SEE_OTHER
        )
        
    except Exception as e:
        print(f"Error al registrar usuario: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al registrar usuario"
        )