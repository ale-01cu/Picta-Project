from fastapi import APIRouter, HTTPException, status, Request, Form
from settings.mongodb import db
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


@router.post("/signup")
async def signup(
    name: str = Form(...), 
    email: str = Form(...), 
    password: str = Form(...)
):
    try:
        # Validación básica de email
        if "@" not in email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Email inválido"
            )

        nuevo_usuario = UserSchema(
            name=name, 
            email=email, 
            password=password
        )
        collection = db["users"]

        # Verificar si el usuario ya existe
        if collection.find_one({"email": email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="El usuario ya existe"
            )
        
        if collection.find_one({"name": name}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="El usuario ya existe"
            )

        collection.insert_one(nuevo_usuario.model_dump())
        return templates.TemplateResponse(
            "signin.html",
            {"request": {}}   
        )
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