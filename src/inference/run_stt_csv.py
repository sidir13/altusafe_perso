# src/inference/run_stt_csv_vosk.py
import os
import csv
import logging
import argparse
import wave
import json
from vosk import Model, KaldiRecognizer
from src.common.config import INFERENCE_DIR, TRANSCRIPTS_DIR, MODELS_DIR

# ---------------------------------------------------------------------
# Parser pour le fichier audio en argument
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Transcrire un fichier audio avec Vosk")
parser.add_argument("audio_file", type=str, help="Chemin complet du fichier audio à transcrire")
args = parser.parse_args()
audio_file = args.audio_file

if not os.path.exists(audio_file):
    raise FileNotFoundError(f"Le fichier audio spécifié n'existe pas : {audio_file}")

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
LOG_PATH = os.path.join(INFERENCE_DIR, "transcription_inference_vosk.log")
os.makedirs(INFERENCE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration des chemins
# ---------------------------------------------------------------------
CSV_PATH = os.path.join(INFERENCE_DIR, "transcription_inference_vosk.csv")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Chargement du modèle Vosk mini
# ---------------------------------------------------------------------
vosk_model_path = os.path.join(MODELS_DIR, "vosk-model-small-fr-0.22")  # <- placez votre modèle mini français ici
if not os.path.exists(vosk_model_path):
    raise FileNotFoundError(f"Le modèle Vosk n'existe pas : {vosk_model_path}")

logger.info(f"🧠 Chargement du modèle Vosk : {vosk_model_path}...")
model = Model(vosk_model_path)
logger.info("✅ Modèle Vosk chargé avec succès !")

# ---------------------------------------------------------------------
# Préparer le chemin de sortie pour la transcription
# ---------------------------------------------------------------------
base_name = os.path.splitext(os.path.basename(audio_file))[0]
transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")

# ---------------------------------------------------------------------
# Ouvrir le fichier audio
# ---------------------------------------------------------------------
wf = wave.open(audio_file, "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 44100]:
    logger.warning("⚠️ Le fichier audio doit être mono PCM 16bit (Vosk peut échouer sinon)")

rec = KaldiRecognizer(model, wf.getframerate())

# ---------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------
logger.info(f"🎧 Transcription en cours : {audio_file}")
transcription_text = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        res = json.loads(rec.Result())
        transcription_text += res.get("text", "") + " "
res = json.loads(rec.FinalResult())
transcription_text += res.get("text", "")

transcription_text = transcription_text.strip()

# ---------------------------------------------------------------------
# Sauvegarder la transcription
# ---------------------------------------------------------------------
with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(transcription_text)
logger.info(f"📝 Transcription sauvegardée : {transcript_path}")

# ---------------------------------------------------------------------
# Ajouter au CSV
# ---------------------------------------------------------------------
csv_exists = os.path.exists(CSV_PATH)
with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(["audio_file", "transcript_file", "transcription_text"])
    writer.writerow([audio_file, transcript_path, transcription_text])
logger.info(f"✅ CSV mis à jour : {CSV_PATH}")
