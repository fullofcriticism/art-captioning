import os
from data_loader import load_classes
from model_utils import load_model, make_prediction, make_description_prediction
from caption_generator import ArtDescriptionGenerator
from text_to_speech import generate_audio
from config import CLASSES_PATH, CLASSIFICATION_MODEL_NAME, DESCRIBING_MODEL_NAME, DEVICE

def main():
    binarizer = load_classes(CLASSES_PATH)
    model, image_shape = load_model(CLASSIFICATION_MODEL_NAME)
    caption_generator = ArtDescriptionGenerator(use_text_model=False)
    
    image_path = "example.jpg" # Введите путь до изображения
    
    predictions = make_prediction(model, binarizer, image_path, image_shape)
    print("Предсказанные классы:", predictions)

    description = caption_generator.generate_description(predictions)
    
    caption = make_description_prediction(DESCRIBING_MODEL_NAME, image_path, DEVICE)
    caption = description + ' ' + caption

    print("Сгенерированное описание:", caption)

    audio_name = "audio_description.mp3" # Введите желаемое название аудио-файла со сгенерированным описанием 
    audio_path = generate_audio(caption, audio_name) 
    print(f"Аудиофайл сохранен как: {audio_path}")

if __name__ == "__main__":
    main()