from typing import Optional
from pydantic import BaseModel

class ImageInput(BaseModel):
    question: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None

class ImageResponse(BaseModel):
    response: str


class TextInput(BaseModel):
    input_string: str


class User(BaseModel):
    username: str
    email: str


