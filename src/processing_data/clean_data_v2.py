import os
import logging
import subprocess
import soundfile as sf
import numpy as np
from src.common.config import MEDECIN_DATA_DIR, PROCESSED_DIR

# --------------------
# Configuration du logger
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("processing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------
# Paramètres
# --------------------
SEGMENT_DURATION = 10  # secondes
OVERLAP = 2            # secondes

# Nouveau sous-dossier pour les segments
SUBDIR = "wav_data_v2"
OUTPUT_DIR = os.path.join(PROCESSED_DIR, SUBDIR)
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"Début du traitement des fichiers audio dans : {MEDECIN_DATA_DIR}")
logger.info(f"Les segments seront enregistrés dans : {OUTPUT_DIR}")

# --------------------
# Fonction conversion universelle (mp3/mp4/m4a/ogg -> wav)
# --------------------
def convert_to_wav(input_path):
    """Convertit un fichier audio en WAV (mono, 16kHz, PCM)."""
    output_name = os.path.splitext(os.path.basename(input_path))[0] + ".wav"
    output_path = os.path.join(TEMP_DIR, output_name)
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", input_path,
            "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
            output_path
        ]
        subprocess.run(cmd, check=True)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"ffmpeg n'a pas généré le fichier de sortie pour {input_path}")

        logger.info(f"Conversion réussie -> {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg a échoué à convertir {input_path} : {e}")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de {input_path} : {e}")

    return None

# --------------------
# Traitement des fichiers
# --------------------
for filename in os.listdir(MEDECIN_DATA_DIR):
    if not filename.lower().endswith((".mp3", ".mp4", ".wav", ".ogg", ".m4a")):
        continue

    filepath = os.path.join(MEDECIN_DATA_DIR, filename)
    logger.info(f"Chargement du fichier : {filename}")

    # 🔹 Convertir si pas déjà en WAV
    if not filename.lower().endswith(".wav"):
        filepath_to_load = convert_to_wav(filepath)
        if not filepath_to_load:
            logger.warning(f"Fichier ignoré (conversion échouée) : {filename}")
            continue
    else:
        filepath_to_load = filepath

    # 🔹 Charger avec soundfile
    try:
        waveform, sample_rate = sf.read(filepath_to_load, dtype="float32")
        if waveform.ndim == 1:  # mono
            waveform = waveform[:, np.newaxis]
    except Exception as e:
        logger.error(f"Impossible de charger {filepath_to_load} : {e}")
        continue

    num_samples = waveform.shape[0]
    duration_sec = num_samples / sample_rate
    logger.info(f"Durée totale : {duration_sec:.2f}s, Sample rate : {sample_rate}, Channels : {waveform.shape[1]}")

    # 🔹 Découpage en segments
    segment_samples = int(SEGMENT_DURATION * sample_rate)
    overlap_samples = int(OVERLAP * sample_rate)

    start = 0
    segment_idx = 0

    while start < num_samples:
        end = min(start + segment_samples, num_samples)
        segment_waveform = waveform[start:end, :]

        segment_name = f"{os.path.splitext(filename)[0]}_seg{segment_idx}.wav"
        out_path = os.path.join(OUTPUT_DIR, segment_name)
        try:
            sf.write(out_path, segment_waveform, sample_rate)
            logger.info(f"  Segment {segment_idx}: {(end - start)/sample_rate:.2f}s -> {segment_name}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du segment {segment_name} : {e}")
            break

        start += segment_samples - overlap_samples
        segment_idx += 1

logger.info(f"Traitement terminé. Tous les segments sont dans : {OUTPUT_DIR}")
