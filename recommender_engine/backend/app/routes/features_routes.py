from engine.data.FeaturesTypes import features_types_map
from fastapi import APIRouter, HTTPException, status, Body, Response

router = APIRouter()

@router.get("/features-types")
async def list_features_types():
    try:
        # Filtrar por configuraciones activas y convertir _id a string
        data = [
            {key: value().label} 
            for key, value in features_types_map.items() 
        ]
        return data
    except Exception as e:
        # Manejar errores de base de datos
        print(f"Error al listar los tipos de caracteristicas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener los tipos de caracteristicas"
        )