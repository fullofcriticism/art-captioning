from datetime import datetime
from pydantic import BaseModel, HttpUrl
from fastapi.responses import FileResponse

class ImageRequest(BaseModel):
    imagePath: str 
    audioName: str
    device: str | None = None
    model_config = {"extra": "forbid"}

class ImageResponse(BaseModel):
    predictedClasses: list[str]
    predictedClassesCaption: str 
    fullDescription: str | None = None
    brailleDescription: str
    audioFile: FileResponse
    audioPath: str