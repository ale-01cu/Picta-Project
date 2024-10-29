from fastapi import APIRouter, Request, Response, HTTPException, status, Query
from engine.actions.ui_actions import (
    train,
    fine_tunning,
    use_engine
)
import time

router = APIRouter()

@router.get("/train/{engine_id}")
async def train_api(engine_id: str):
    try:
        train.train(engine_id)
        return Response(
            None, 
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Train Failed"
        ) 

@router.get("/tunning/{engine_id}")
async def fine_tunning_api(engine_id: str):
    # try:
    fine_tunning.fine_tunning(engine_id)
    return Response(
        None,
        status_code=status.HTTP_200_OK
    )
    
    # except Exception as e:
    #     print(e)
    #     return HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Tunning Failed"
    #     )

@router.get("/recommend/{user_id}")
async def recommend_api(user_id: int, request: Request, k: int = Query(10)):
    params = request.query_params._dict
    start_time = time.time()
    try:
        recommendations = use_engine.use_models(user_id, k, params)
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
    
        