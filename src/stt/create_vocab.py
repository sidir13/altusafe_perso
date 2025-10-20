# src/stt/create_vocab_medical.py
import os
import json
import logging
from src.common.config import TRANSCRIPTS_DIR, RESULTS_DIR

# --------------------
# Configuration
# --------------------
VOCAB_DATA_DIR = os.path.join(RESULTS_DIR, "vocab_data")
os.makedirs(VOCAB_DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(VOCAB_DATA_DIR, "create_vocab_medical.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------
# Stopwords français (liste minimaliste)
# --------------------
STOPWORDS = {
    "le", "la", "les", "un", "une", "et", "de", "des", "du", "dans", "sur",
    "à", "pour", "est", "avec", "au", "aux", "ce", "ces", "il", "elle", "on",
    "ne", "pas", "que", "qui", "se", "sa", "son", "sont", "comme", "ou", "par"
}

# --------------------
# Extraction du vocabulaire
# --------------------
vocab_set = set()

for filename in os.listdir(TRANSCRIPTS_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(TRANSCRIPTS_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                words = [w.strip().lower() for w in text.replace("\n", " ").split() if w.strip()]
                # On garde les mots qui ne sont pas des stopwords et qui ne sont pas des chiffres
                medical_words = [w for w in words if w not in STOPWORDS and not w.isdigit()]
                vocab_set.update(medical_words)
        except Exception as e:
            logger.warning(f"Impossible de lire {file_path} : {e}")

vocab_list = sorted(vocab_set)
output_path = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")

try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulaire médical généré ({len(vocab_list)} mots) -> {output_path}")
except Exception as e:
    logger.error(f"Impossible d'enregistrer le vocabulaire : {e}")
