from gtts import gTTS
import os
from config import AUDIO_OUTPUT_DIR, AUDIO_VOICE, AUDIO_SPEED

def generate_audio(text, filename="output.mp3"):
    """Генерация аудиофайла из текста на русском языке"""
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    
    tts = gTTS(text=text, lang=AUDIO_VOICE, slow=False)
    tts.speed = AUDIO_SPEED
    
    output_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
    tts.save(output_path)
    return output_path