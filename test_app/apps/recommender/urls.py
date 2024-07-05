from .views import GetRecommendations
from django.urls import path

urlpatterns = [
    path("rs/<int:k>/", GetRecommendations.as_view())
]