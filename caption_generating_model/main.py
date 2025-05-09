import os
from data_loader import load_classes
from model_utils import load_model, make_prediction, make_description_prediction, make_translation
from caption_generator import ArtDescriptionGenerator
from text_to_speech import generate_audio
from config import CLASSES_PATH, CLASSIFICATION_MODEL_NAME, DESCRIBING_MODEL_NAME,  TRANSLATION_MODEL_NAME, DEVICE, TRANSLATION_TOKEN_ID

def main():
    binarizer = load_classes(CLASSES_PATH)
    model, image_shape = load_model(CLASSIFICATION_MODEL_NAME)
    caption_generator = ArtDescriptionGenerator(use_text_model=False)
    
    image_path = "example.jpg" # Введите путь до изображения
    
    predictions = make_prediction(model, binarizer, image_path, image_shape)
    print("Предсказанные классы:", predictions)

    description = caption_generator.generate_description(predictions)
    print("Описание по предсказанным классам:", description)
    
    if DEVICE: 
        english_caption = make_description_prediction(DESCRIBING_MODEL_NAME, image_path, DEVICE)
        caption = make_translation(TRANSLATION_MODEL_NAME, english_caption, TRANSLATION_TOKEN_ID)
        print("Сгенерированное описание:", caption)
    else: 
        caption = ''
        print("Нет доступа к GPU, описание объектов на изображении не доступно.")
    
    description += ' ' + caption

    audio_name = "audio_description.mp3" # Введите желаемое название аудио-файла со сгенерированным описанием 
    audio_path = generate_audio(description, audio_name) 
    print(f"Аудиофайл сохранен как: {audio_path}")

if __name__ == "__main__":
    main()