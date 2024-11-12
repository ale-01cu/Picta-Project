from pydantic import BaseModel
from typing import Optional

class PlataformSchema(BaseModel):
  name: str
  secret_key: str