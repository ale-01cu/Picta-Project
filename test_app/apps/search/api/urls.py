from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ContenidoView, BusquedaView

router = DefaultRouter()
router.register('contenido', ContenidoView, basename='contenido')
router.register(r'search(?P<busqueda>)', BusquedaView, basename='busqueda')

urlpatterns = router.urls
