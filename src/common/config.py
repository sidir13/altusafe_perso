"""
config.py
---------
Fichier de configuration centralisée pour les chemins du projet STT Vosk.
Importe ces variables dans tous tes scripts pour éviter les chemins relatifs.
"""

import os

# ---------------------------------------------------------------------
#  Dossiers principaux
# ---------------------------------------------------------------------

# Racine du projet (le dossier "altusafe")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dossier de reporting
REPORTING_DIR = os.path.join(BASE_DIR, "data", "reporting")
os.makedirs(REPORTING_DIR, exist_ok=True)  # crée le dossier si nécessaire

# Sous-dossiers
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PROCESSED_DIR = os.path.join(DATA_DIR,"processed")
# ---------------------------------------------------------------------
# Sous-dossiers des données
# ---------------------------------------------------------------------

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MEDECIN_DATA_DIR = os.path.join(RAW_DATA_DIR, "enregistrements")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VOCAB_DATA_DIR = os.path.join(DATA_DIR, "vocabulaire")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
TSV_DIR = os.path.join(DATA_DIR, "tsv")  

# ---------------------------------------------------------------------
#  Fichiers résultats
# ---------------------------------------------------------------------

# BENCHMARK_CSV = os.path.join(RESULTS_DIR, "benchmarks.csv")
WER_CSV = os.path.join(RESULTS_DIR, "wer_scores.csv")

# ---------------------------------------------------------------------
# Autres constantes utiles
# ---------------------------------------------------------------------

SAMPLE_RATE = 16000  # fréquence d’échantillonnage standard
DEFAULT_MODEL_FR = os.path.join(MODELS_DIR, "vosk-model-small-fr-0.22")
