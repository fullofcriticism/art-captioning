import tensorflow as tf
from skimage import io
from skimage.transform import resize
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
from PIL import Image

def load_classes_model(model_name):
    model = tf.keras.models.load_model(model_name)
    image_shape = (299, 299, 3)
    return model, image_shape

def make_classes_prediction(model, binarizer, image_path, image_shape):
    image = resize(io.imread(image_path), image_shape)
    image = image.reshape((-1,) + image_shape)
    y_pred = model.predict(image).round()
    return list(binarizer.inverse_transform(y_pred)[0])


def make_description_prediction(model_id, image_path, device):
    image = Image.open(image_path).convert("RGB")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device) # stopped working in transformers 4.50.0, still works in 4.49.0. gpu is required
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.3,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=prompt,
        image_size=(image.width, image.height)
    )

    parsed_answer = parsed_answer[prompt]

    return parsed_answer.capitalize()

def make_translation(model_name, text, device):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translation_model.to(device)
    encoded_descr = tokenizer(text, return_tensors="pt")
    translated_tokens = translation_model.generate(**encoded_descr.to(device), forced_bos_token_id=tokenizer.convert_tokens_to_ids("rus_Cyrl"))

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

