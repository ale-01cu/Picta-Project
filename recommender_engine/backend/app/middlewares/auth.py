from fastapi import Request, status
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
                response = templates.TemplateResponse("403.html", {"request": request})
                await response(scope, receive, send)
                return
            
            try:
                # Decodificar el token JWT
                jwt.decode(
                    auth_token, 
                    SECRET_KEY, 
                    algorithms=["HS256"]
                )
                # Puedes acceder a los datos del token si es necesario, por ejemplo:
                # username = payload.get("sub")
            except jwt.ExpiredSignatureError:
                response = templates.TemplateResponse("403.html", {"request": request})
                await response(scope, receive, send)
                return
            except jwt.InvalidTokenError:
                response = templates.TemplateResponse("403.html", {"request": request})
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)
