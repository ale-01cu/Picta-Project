from fastapi import APIRouter, HTTPException, status, Request, Form, Response
from settings.mongodb import db
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.schemas.UserSchema import UserSchema, UserSignInSchema

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

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
    request: Request,
    email: str = Form(...), 
    password: str = Form(...)
):
    try:
        collection = db["users"]
        
        # Buscar el usuario por email
        user = collection.find_one({"email": email})
        
        if not user or user["password"] != password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Credenciales incorrectas"
            )
        
        # Crear un token simple
        token = "usuario_autenticado"
        
        # Crear una respuesta de redirección
        response = RedirectResponse(
            "/config",
            status_code=status.HTTP_303_SEE_OTHER
        )
        
        # Establecer la cookie con el token
        response.set_cookie(key="auth_token", value=token, httponly=True)
        
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
        form_fields = UserSignInSchema.model_fields.keys()
        return templates.TemplateResponse(
            "signin.html", 
            {"request": request, "form_fields": form_fields}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )


@router.post("/signin")
async def signin(
    request: Request,
    email: str = Form(...), 
    password: str = Form(...)
):
    try:
        collection = db["users"]
        
        # Buscar el usuario por email
        user = collection.find_one({"email": email})
        
        if not user or user["password"] != password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Credenciales incorrectas"
            )
        
        request.session['user'] = user['email']
        # Aquí podrías agregar lógica para crear una sesión o token
        
        return templates.TemplateResponse(
            "home.html", 
            {"request": {}}
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )