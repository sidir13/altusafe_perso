"""
vocab_injector.py
-----------------
Reconnaissance vocale utilisant un modèle Vosk enrichi
avec un vocabulaire médical personnalisé.
"""

import os
import json
import wave
from vosk import Model, KaldiRecognizer

# 🧩 Import de la configuration centralisée
from src.common.config import (
    EXPERIMENTAL_MODEL_FR,
    WAV_DATA_DIR,
    TRANSCRIPTS_DIR,
    SAMPLE_RATE,
    MEDICAL_VOCABULARY
)

# ---------------------------------------------------------------------
# Fonctions principales
# ---------------------------------------------------------------------

def recognize_with_medical_vocab(audio_path, model):
    """
    Reconnaît un fichier audio avec le vocabulaire médical.
    Retourne le texte reconnu.
    """
    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(MEDICAL_VOCABULARY))

    with wave.open(audio_path, "rb") as wf:
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "")


def process_all_wav_files(wav_dir=WAV_DATA_DIR, output_dir=TRANSCRIPTS_DIR):
    """
    Parcourt tous les fichiers .wav dans WAV_DATA_DIR,
    applique la reconnaissance vocale médicale et sauvegarde les résultats.
    """
    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"Le dossier audio '{wav_dir}' est introuvable.")

    os.makedirs(output_dir, exist_ok=True)

    # Chargement du modèle expérimental
    print(f"Chargement du modèle expérimental : {EXPERIMENTAL_MODEL_FR}")
    model = Model(EXPERIMENTAL_MODEL_FR)

    results = {}

    for filename in os.listdir(wav_dir):
        if filename.lower().endswith(".wav"):
            audio_path = os.path.join(wav_dir, filename)
            print(f"\n🩺 Traitement de {filename} ...")
            text = recognize_with_medical_vocab(audio_path, model)
            results[filename] = text
            print(f"→ Reconnu : {text}")

    #  Sauvegarde des transcriptions dans un JSON
    output_file = os.path.join(output_dir, "medical_transcripts.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n Résultats enregistrés dans : {output_file}")


# ---------------------------------------------------------------------
# Exécution principale
# ---------------------------------------------------------------------
if __name__ == "__main__":
    process_all_wav_files()
