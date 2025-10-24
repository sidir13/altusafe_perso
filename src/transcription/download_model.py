import os
import whisper
from src.common.config import MODELS_DIR, PROCESSED_DIR

TRANSCRIPT_DIR = os.path.join("data", "transcripts")
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Chemin pour le cache du modèle
os.environ["TORCH_HOME"] = MODELS_DIR

# Charger un modèle Whisper open source
model = whisper.load_model("medium")  
print(f" Modèle Whisper chargé depuis {MODELS_DIR}")

# Parcourir les fichiers audio
for file in os.listdir(PROCESSED_DIR):
    if file.endswith(".wav"):
        audio_path = os.path.join(PROCESSED_DIR, file)
        print(f" Transcription de : {audio_path}")

        result = model.transcribe(audio_path, language="fr")

        # Sauvegarder la transcription
        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{os.path.splitext(file)[0]}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        print(f"Transcription enregistrée : {transcript_path}")

print("\nToutes les transcriptions sont terminées !")
