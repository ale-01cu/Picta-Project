from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive=receive)
        
        # Rutas permitidas sin autenticación
        allowed_paths = ["/signin", "/signup"]
        
        if request.url.path not in allowed_paths:
            auth_token = request.cookies.get("auth_token")
            if auth_token != "usuario_autenticado":
                response = templates.TemplateResponse(
                    "403.html", {"request": request})
                await response(scope, receive, send)
                return
            

        await self.app(scope, receive, send)