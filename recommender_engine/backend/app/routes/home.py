from fastapi import (
    APIRouter, 
    Request, 
)
from fastapi.templating import Jinja2Templates
from engine.data.DataPipeline import DataPipeline

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/")
async def home(request: Request):
    # try:
    pipe = DataPipeline(logging=False)
    users_df, = pipe.read_csv_data(
        paths=['../../datasets/usuarios_timestamp.csv'])
    
    # Filtrar por configuraciones activas y convertir _id a string
    return templates.TemplateResponse(
        "home.html", 
        {"request": request, "users": users_df.head(10).to_dict(orient='records')}
    )
    # except Exception as e:
    #     # Manejar errores de base de datos
    #     print(f"Error al acceder a la pagina de inicio: {e}")
    #     return templates.TemplateResponse(
    #         "500.html", 
    #         {"request": request}
    #     )
    
@router.get("/{user_id}")
async def home_user_selected(user_id: int, request: Request):
    try:
        pipe = DataPipeline(logging=False)
        users_df, views_df, likes_df = pipe.read_csv_data(paths=[
            '../../datasets/usuarios_timestamp.csv',
            '../../datasets/vistas.csv',
            '../../datasets/likes.csv',
        ])

        print(views_df)

        views_result = views_df.loc[views_df['usuario_id'] == user_id]
        likes_result = likes_df.loc[likes_df['usuario_id'] == user_id]
        user = users_df.drop_duplicates(subset='id').loc[users_df['id'] == user_id]

        print(views_result)

        # Filtrar por configuraciones activas y convertir _id a string
        return templates.TemplateResponse(
            "home.html", 
            {
                "request": request, 
                "users": users_df.tail(10).to_dict(orient='records'), 
                "user": list(user.to_dict()['username'].values())[0],
                "views": views_result.head(10).to_dict(orient='records'), 
                "likes": likes_result.head(10).to_dict(orient='records')
            }
        )
    except Exception as e:
        # Manejar errores de base de datos
        print(f"Error al listar las preferencias de los usuarios en la pagina inicio: {e}")
        return templates.TemplateResponse(
            "500.html", 
            {"request": request}
        )
    

# @router.get("/{user_id}/results")
# async def home_with_recommendations(user_id: int, request: Request):
#     try:
#         pipe = DataPipeline(logging=False)
#         users_df, views_df, likes_df = pipe.read_csv_data(paths=[
#             '../../datasets/usuarios_timestamp.csv',
#             '../../datasets/vistas.csv',
#             '../../datasets/likes.csv',
#         ])

#         views_result = views_df.loc[views_df['usuario_id'] == user_id]
#         likes_result = likes_df.loc[likes_df['usuario_id'] == user_id]
#         user = users_df.loc[users_df['id'] == user_id]
        


#         # Filtrar por configuraciones activas y convertir _id a string
#         return templates.TemplateResponse(
#             "home.html", 
#             {
#                 "request": request, 
#                 "users": users_df, 
#                 "user": user,
#                 "views": views_result, 
#                 "likes": likes_result,
#                 "recommendations": []
#             }
#         )
#     except Exception as e:
#         # Manejar errores de base de datos
#         print(f"Error al listar los tipos de caracteristicas: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error al obtener los tipos de caracteristicas"
#         )