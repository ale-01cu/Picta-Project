from fastapi import Request, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import jwt

templates = Jinja2Templates(directory="app/templates")

class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from main import SECRET_KEY
        request = Request(scope, receive=receive)
        
        # Rutas permitidas sin autenticación
        allowed_paths = ["/signin", "/signup"]
        
        if request.url.path not in allowed_paths:
            auth_token = request.cookies.get("auth_token")
            if not auth_token:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Token de autenticación no encontrado"
                )
            
            try:
                # Decodificar el token JWT
                payload = jwt.decode(auth_token, SECRET_KEY, algorithms=["HS256"])
                # Puedes acceder a los datos del token si es necesario, por ejemplo:
                # username = payload.get("sub")
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="El token ha expirado"
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token inválido"
                )

        await self.app(scope, receive, send)