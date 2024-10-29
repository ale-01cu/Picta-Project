from pydantic import BaseModel

class UserSchema(BaseModel):
    name: str
    email: str
    password: str

class UserSignInSchema(BaseModel):
    email: str
    password: str