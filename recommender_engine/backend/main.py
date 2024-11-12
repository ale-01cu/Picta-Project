from fastapi import FastAPI, Request
from app.routes import (
    config_routes,
    features_routes,
    auth_routes,
    engine_routes,
    engine_actions_routes,
    authorizate_plataforms
)
from app.middlewares.auth import AuthMiddleware
from fastapi.templating import Jinja2Templates

# build_db()
app = FastAPI()

app.add_middleware(AuthMiddleware)

app.include_router(config_routes.router)
app.include_router(features_routes.router)
app.include_router(auth_routes.router)
app.include_router(engine_routes.router)
app.include_router(engine_actions_routes.router)
app.include_router(authorizate_plataforms.router)

SECRET_KEY = "tu_secreto_super_secreto"


templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "home.html", 
        {"request": request}
    )