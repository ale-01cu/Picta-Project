# Create your models here.
from django.db import models
import json

# Create your models here.
class Contenido(models.Model):
  choices = [
    ('Pelicula', 'Pelicula'),
    ('Serie', 'Serie'),
    ('Novela', 'Novela')
  ]
  
  categoria = models.CharField(max_length=50, choices=choices)
  titulo = models.CharField(max_length=200)
  fecha_de_estreno = models.DateField()
  descripcion = models.TextField()
  generos = models.CharField(max_length=200)
  
  def __str__(self):
    return self.categoria + " " + self.titulo
  
  def set_genero (self, lst):
        self.generos = json.dumps(lst)

  def get_genero (self):
      return json.loads(self.generos)