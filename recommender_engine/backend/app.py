from fastapi import FastAPI, Query, HTTPException, status, Response, Request
from settings.db import get_db
from sqlalchemy.orm import Session
from engine.db.main import build_db
from engine.actions.use_engine import use_models
from engine.actions.train import train
import time

build_db()
app = FastAPI()

@app.get("/train")
async def train_api():
    try:
        train()
        return Response(
            None, 
            status_code=status.HTTP_201_CREATED
        )
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Train Failed"
        ) 


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
        