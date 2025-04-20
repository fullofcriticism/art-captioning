from datetime import datetime
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse

class Request(BaseModel):
    imagePath: str 
    audioName: str
    device: str | None = None
    model_config = {"extra": "forbid"}

class Response(BaseModel):
    predictedClasses: list[str]
    predictedClassesCaption: str 
    fullDescription: str | None = None
    audioFile: FileResponse
    audioPath: str

class TextRequest(BaseModel):
    text: str 
    model_config = {"extra": "forbid"}
