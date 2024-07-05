from django.apps import AppConfig
from .tasks import read_dataset

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.recommender'

    # def ready(self):
    #     # Ejecutar script aquí
    #     read_dataset()