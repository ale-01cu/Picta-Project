from django.db import models
from database.crud import DatabaseModelBase
# Create your models here.

class PubsModel(DatabaseModelBase):
    db_filename = "pubs.csv"


class ClicksModel(DatabaseModelBase):
    db_filename = "clicks.csv" 