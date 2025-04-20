from fastapi import FastAPI 
from fastapi.responses import JSONResponse, FileResponse
import torch
from PIL import Image
import requests
import os
from data_loader import load_classes
from model_utils import load_model, make_prediction, make_description_prediction, make_translation
from caption_generator import ArtDescriptionGenerator
from text_to_speech import generate_audio
from config import CLASSES_PATH, CLASSIFICATION_MODEL_NAME, DESCRIBING_MODEL_NAME,  TRANSLATION_MODEL_NAME
from http_models import Request, Response


app = FastAPI()

binarizer = load_classes(CLASSES_PATH)
model, image_shape = load_model(CLASSIFICATION_MODEL_NAME)
caption_generator = ArtDescriptionGenerator(use_text_model=False)


@app.post("/predict/classes", response_model=Response)
async def predict_classes(body: Request):
    file_path = body.filePath
    predictions = make_prediction(model, binarizer, file_path, image_shape)
    return JSONResponse(content={"predictedClasses": predictions})

@app.post("/generate/caption", response_model=Response)
async def generate_caption(body: Request):
    file_path = body.imagePath
    device = body.device
    audio_name = body.audioName

    predictions = make_prediction(model, binarizer, file_path, image_shape)
    description = caption_generator.generate_description(predictions)

    if device: 
        english_caption = make_description_prediction(DESCRIBING_MODEL_NAME, file_path, device)
        caption = make_translation(TRANSLATION_MODEL_NAME, english_caption, device)
    else: 
        caption = ''

    description += ' ' + caption
    
    audio_path = generate_audio(caption, audio_name)
    
    return JSONResponse(content={"predictedClasses": predictions, 
                                 "predictedClassesCaption": description, 
                                 "fullDescription": caption, 
                                 "audioFile": FileResponse(audio_path),
                                 "audioPath": audio_path})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
