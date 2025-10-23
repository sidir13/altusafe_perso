import json
import os
from src.common.config import VOCAB_DATA_DIR  # chemin centralisé

# Nom du fichier vocab
vocab_file = "medical_vocab_filtered.json"

# Chemin complet vers le vocab
vocab_path = os.path.join(VOCAB_DATA_DIR, vocab_file)

# Charger le vocabulaire
try:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)  # vocab attendu comme liste de mots
except FileNotFoundError:
    raise FileNotFoundError(f"Fichier vocab non trouvé : {vocab_path}")

if not vocab:
    raise ValueError(f"Vocabulaire vide dans {vocab_path}")

# Créer dictionnaire phonétique (sans phonemizer)
vocab_phon = {w: w for w in vocab}  # mot lui-même comme “phonétique”

# Sauvegarder dictionnaire phonétique
phon_file = "medical_vocab_phon.json"
phon_path = os.path.join(VOCAB_DATA_DIR, phon_file)
with open(phon_path, "w", encoding="utf-8") as f:
    json.dump(vocab_phon, f, ensure_ascii=False, indent=2)

print(f"✅ Dictionnaire phonétique sauvegardé dans {phon_path}")
