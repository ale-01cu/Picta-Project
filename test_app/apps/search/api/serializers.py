from rest_framework import serializers
from App_Buscador.models import Contenido

class SerializerContenido(serializers.ModelSerializer):
  class Meta():
    model = Contenido
    fields = '__all__'