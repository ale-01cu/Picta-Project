from django_elasticsearch_dsl import Document
from django_elasticsearch_dsl.registries import registry
from App_Buscador.models import Contenido
from django_elasticsearch_dsl import Document, fields


@registry.register_document
class ContenidoDocument(Document):
    class Index:
        name = 'contenido'

    id = fields.IntegerField(attr='id')
    titulo = fields.TextField(analyzer='spanish', fielddata=True)
    categoria = fields.TextField(analyzer='spanish')
    generos = fields.TextField(analyzer='spanish')
    descripcion = fields.TextField(analyzer='spanish')

    class Django:
        model = Contenido
