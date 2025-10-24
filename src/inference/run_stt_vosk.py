import os
import csv
import json
import wave
import logging
import argparse
from vosk import Model, KaldiRecognizer
import Levenshtein  # pip install python-Levenshtein

from src.common.config import INFERENCE_DIR, TRANSCRIPTS_DIR, MODELS_DIR, VOCAB_DATA_DIR
from src.nlp.medical_postprocessor import MedicalPostProcessorPhonetic

# ---------------------------------------------------------------------
# Parser pour le dossier ou fichier audio
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Transcrire des fichiers audio avec Vosk + post-traitement médical")
parser.add_argument("audio_path", type=str, help="Chemin vers le fichier audio ou le dossier contenant des .wav")
args = parser.parse_args()
audio_path = args.audio_path

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Le chemin spécifié n'existe pas : {audio_path}")

# ---------------------------------------------------------------------
# Config / Logs
# ---------------------------------------------------------------------
os.makedirs(INFERENCE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

LOG_PATH = os.path.join(INFERENCE_DIR, "transcription_inference_vosk.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------
CSV_PATH = os.path.join(INFERENCE_DIR, "transcription_inference_vosk.csv")
vosk_model_path = os.path.join(MODELS_DIR, "vosk-model-small-fr-0.22")
vocab_path = os.path.join(VOCAB_DATA_DIR, "medical_vocab_phon.json")

# ---------------------------------------------------------------------
# Chargement du modèle Vosk
# ---------------------------------------------------------------------
logger.info(f"Chargement du modèle Vosk : {vosk_model_path}")
model = Model(vosk_model_path)

# ---------------------------------------------------------------------
# Chargement du post-traitement médical phonétique
# ---------------------------------------------------------------------
processor = MedicalPostProcessorPhonetic(vocab_json_path=vocab_path, threshold=0.7, top_n=5)

# ---------------------------------------------------------------------
# Liste des fichiers audio à traiter
# ---------------------------------------------------------------------
if os.path.isfile(audio_path) and audio_path.endswith(".wav"):
    audio_files = [audio_path]
elif os.path.isdir(audio_path):
    audio_files = [os.path.join(audio_path, f) for f in os.listdir(audio_path) if f.endswith(".wav")]
else:
    raise ValueError("Le chemin spécifié n'est pas un fichier .wav ou un dossier valide")

# ---------------------------------------------------------------------
# CSV : création si nécessaire
# ---------------------------------------------------------------------
csv_exists = os.path.exists(CSV_PATH)
with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow([
            "audio_file",
            "transcript_file_brut",
            "transcription_brute",
            "transcript_file_corrige",
            "transcription_corrigee",
            "levenshtein_distance",
            "levenshtein_distance_normalized",
            "replacements",
            "cosine_scores"
        ])

# ---------------------------------------------------------------------
# Boucle sur les fichiers audio
# ---------------------------------------------------------------------
for audio_file in audio_files:
    logger.info(f"Transcription de {audio_file}")

    # Transcription brute avec Vosk
    wf = wave.open(audio_file, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += res.get("text", "") + " "
    res = json.loads(rec.FinalResult())
    text += res.get("text", "")
    text = text.strip()

    # Sauvegarde de la transcription brute
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    transcript_path_brut = os.path.join(TRANSCRIPTS_DIR, f"{base_name}_brut.txt")
    with open(transcript_path_brut, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Transcription brute sauvegardée : {transcript_path_brut}")

    # Post-traitement médical contextuel
    corrected_text, replacements, cosine_scores = processor.process_sentence(text)
    transcript_path_corrige = os.path.join(TRANSCRIPTS_DIR, f"{base_name}_corrige.txt")
    with open(transcript_path_corrige, "w", encoding="utf-8") as f:
        f.write(corrected_text)
    logger.info(f"Transcription corrigée sauvegardée : {transcript_path_corrige}")

    # Calcul de la distance de Levenshtein
    lev_dist = Levenshtein.distance(text, corrected_text)
    lev_dist_norm = lev_dist / max(len(text), 1)

    # Conversion JSON-safe
    cosine_scores = {k: float(v) for k, v in cosine_scores.items()}

    # Mise à jour du CSV
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            audio_file,
            transcript_path_brut,
            text,
            transcript_path_corrige,
            corrected_text,
            lev_dist,
            lev_dist_norm,
            json.dumps(replacements, ensure_ascii=False),
            json.dumps(cosine_scores, ensure_ascii=False)
        ])

    logger.info(f"CSV mis à jour : {CSV_PATH}")
