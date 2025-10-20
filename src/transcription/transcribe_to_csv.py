import os
import csv
import whisper
import logging
from src.common.config import MODELS_DIR, PROCESSED_DIR, TRANSCRIPTS_DIR, RESULTS_DIR

# ---------------------------------------------------------------------
# Configuration du logger
# ---------------------------------------------------------------------
LOG_PATH = os.path.join(RESULTS_DIR, "transcription.log")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration des chemins
# ---------------------------------------------------------------------
os.environ["TORCH_HOME"] = MODELS_DIR
CSV_PATH = os.path.join(RESULTS_DIR, "transcriptions.csv")

os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Chargement du modèle Whisper
# ---------------------------------------------------------------------
logger.info("🧠 Chargement du modèle Whisper (medium)...")
try:
    model = whisper.load_model("medium")  # medium = bon équilibre vitesse/qualité
    logger.info("✅ Modèle Whisper chargé avec succès !")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle Whisper : {e}")
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Création / ouverture du CSV
# ---------------------------------------------------------------------
csv_exists = os.path.exists(CSV_PATH)
try:
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        # Écrire l'en-tête si le fichier est nouveau
        if not csv_exists:
            writer.writerow(["audio_file", "transcript_file", "transcription_text"])
            logger.info("📄 Nouveau fichier CSV créé avec en-tête.")

        # -----------------------------------------------------------------
        # Parcourir les fichiers audio
        # -----------------------------------------------------------------
        for file in os.listdir(PROCESSED_DIR):
            if not file.endswith(".wav"):
                continue

            audio_path = os.path.join(PROCESSED_DIR, file)
            base_name = os.path.splitext(file)[0]
            transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")

            logger.info(f"🎧 Transcription en cours : {audio_path}")
            try:
                result = model.transcribe(audio_path, language="fr")
                text = result["text"].strip()
            except Exception as e:
                logger.error(f"Erreur pendant la transcription de {file} : {e}")
                continue

            # # Sauvegarder la transcription dans un fichier texte
            # try:
            #     with open(transcript_path, "w", encoding="utf-8") as f:
            #         f.write(text)
            #     logger.info(f"📝 Transcription sauvegardée : {transcript_path}")
            # except Exception as e:
            #     logger.error(f"Impossible d’enregistrer {transcript_path} : {e}")
            #     continue

            # Ajouter au CSV
            writer.writerow([audio_path, transcript_path, text])
            logger.info(f"✅ Ligne ajoutée au CSV pour {file}")

    logger.info(f"\n📁 Toutes les transcriptions sont enregistrées dans : {CSV_PATH}")

except Exception as e:
    logger.error(f"Erreur générale pendant la création du CSV : {e}")
