from fastapi import FastAPI, Body, Query, HTTPException, status, Response, Request
from engine.actions.ui_actions.use_engine import use_models
from engine.actions.ui_actions.train import train
from app.routes import (
    config_routes,
    features_routes,
    auth_routes,
    engine_routes
)
import time

# build_db()
app = FastAPI()



app.include_router(config_routes.router)
app.include_router(features_routes.router)
app.include_router(auth_routes.router)
app.include_router(engine_routes.router)


@app.get("/train/{engine_id}")
async def train_api(engine_id: str):
    # try:
    train(engine_id)
    return Response(
        None, 
        status_code=status.HTTP_200_OK
    )
    # except Exception as e:
    #     print(e)
    #     return HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Train Failed"
    #     ) 


@app.get("/recommend/{user_id}")
async def recommend_api(user_id: int, request: Request, k: int = Query(10)):
    params = request.query_params._dict
    start_time = time.time()
    try:
        recommendations = use_models(user_id, k, params)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        return recommendations
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommend Failed"
        )
        