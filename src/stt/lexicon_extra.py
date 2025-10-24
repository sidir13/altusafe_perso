import os
import json
import logging
from src.common.config import VOCAB_DATA_DIR

# ------------------ Configuration du logger ------------------
LOG_PATH = os.path.join(VOCAB_DATA_DIR, "generate_lexicon.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ Fichiers ------------------
LEXICON_PATH = os.path.join(VOCAB_DATA_DIR, "medical_vocab_phon.json")
OUTPUT_LEXIC = os.path.join(VOCAB_DATA_DIR, "lexicon_extra.txt")

# ------------------ Script principal ------------------
def main():
    logger.info("🚀 Début de la génération du lexique phonétique...")

    # Vérification d'existence
    if not os.path.exists(LEXICON_PATH):
        logger.error(f"❌ Fichier introuvable : {LEXICON_PATH}")
        return

    try:
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            lexicon_extra = json.load(f)
        logger.info(f"✅ Chargement réussi : {len(lexicon_extra)} entrées trouvées.")
    except json.JSONDecodeError as e:
        logger.exception(f"Erreur lors du chargement du JSON : {e}")
        return
    except Exception as e:
        logger.exception(f"Erreur inattendue pendant la lecture : {e}")
        return

    try:
        count = 0
        with open(OUTPUT_LEXIC, "w", encoding="utf-8") as f:
            for word, ph in lexicon_extra.items():
                if not word or not ph:
                    logger.warning(f"⚠️ Entrée incomplète ignorée : {word} -> {ph}")
                    continue
                phones = " ".join(ph.replace(".", " ").split())
                f.write(f"{word} {phones}\n")
                count += 1

        logger.info(f"✅ Lexique généré : {count} mots écrits dans {OUTPUT_LEXIC}")

    except Exception as e:
        logger.exception(f"Erreur lors de l'écriture du fichier : {e}")
        return

    logger.info("🏁 Fin du processus de génération du lexique phonétique.")

# ------------------ Exécution ------------------
if __name__ == "__main__":
    main()
