from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import torch
from PIL import Image
import requests
import os
from data_loader import load_classes
from model_utils import load_model, make_prediction, make_description_prediction, make_translation
from caption_generator import ArtDescriptionGenerator
from text_to_speech import generate_audio
from config import CLASSES_PATH, CLASSIFICATION_MODEL_NAME, DESCRIBING_MODEL_NAME,  TRANSLATION_MODEL_NAME
from http_models import ImageRequest, ImageResponse


app = FastAPI()

binarizer = load_classes(CLASSES_PATH)
model, image_shape = load_model(CLASSIFICATION_MODEL_NAME)
caption_generator = ArtDescriptionGenerator(use_text_model=False)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"errorCode": exc.errors(), 
                                  "errorMessage": exc.body}),
        )

@app.post("/predict/classes", response_model=ImageResponse)
async def predict_classes(body: ImageRequest):
    file_path = body.filePath
    predictions = make_prediction(model, binarizer, file_path, image_shape)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({"predictedClasses": predictions})
        )

@app.post("/generate/caption", response_model=ImageResponse)
async def generate_caption(body: Request):
    file_path = body.imagePath
    device = body.device
    audio_name = body.audioName

    predictions = make_prediction(model, binarizer, file_path, image_shape)
    description = caption_generator.generate_description(predictions)
    try: 
        if device: 
            english_caption = make_description_prediction(DESCRIBING_MODEL_NAME, file_path, device)
            caption = make_translation(TRANSLATION_MODEL_NAME, english_caption, device)
        else: 
            caption = ''
        description += ' ' + caption
    
        audio_path = generate_audio(caption, audio_name)
    
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=jsonable_encoder({"predictedClasses": predictions, 
                                 "predictedClassesCaption": description, 
                                 "fullDescription": caption, 
                                 "audioFile": FileResponse(audio_path),
                                 "audioPath": audio_path})
            )
    except:
        return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"errorCode": 'UnprocessableEntity', 
                                  "errorMessage": 'Error occured and request cannot be processed'}),
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
