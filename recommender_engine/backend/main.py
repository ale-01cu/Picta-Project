from fastapi import FastAPI
from app.routes import (
    config_routes,
    features_routes,
    auth_routes,
    engine_routes,
    engine_actions_routes
)
from app.middlewares.auth import AuthMiddleware

# build_db()
app = FastAPI()

app.add_middleware(AuthMiddleware)

app.include_router(config_routes.router)
app.include_router(features_routes.router)
app.include_router(auth_routes.router)
app.include_router(engine_routes.router)
app.include_router(engine_actions_routes.router)


