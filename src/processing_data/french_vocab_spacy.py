import os
import json
import spacy
import re
from src.common.config import VOCAB_DATA_DIR

# -----------------------------
# Chemins
# -----------------------------
MEDICAL_VOCAB_PATH = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")
MERGED_VOCAB_PATH = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")  # On écrase le fichier
os.makedirs(VOCAB_DATA_DIR, exist_ok=True)

# -----------------------------
# Charger le vocabulaire médical existant
# -----------------------------
if os.path.exists(MEDICAL_VOCAB_PATH):
    with open(MEDICAL_VOCAB_PATH, "r", encoding="utf-8") as f:
        medical_vocab = set(json.load(f))
else:
    medical_vocab = set()

print(f"Vocabulaire médical existant : {len(medical_vocab)} mots")

# -----------------------------
# Extraire le vocabulaire français courant depuis SpaCy
# -----------------------------
nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

common_french_words = {
    lex.text.lower()
    for lex in nlp.vocab
    if lex.is_alpha and lex.text.isascii() and len(lex.text) > 2 and lex.text.lower() not in stopwords
}

print(f"Vocabulaire français courant SpaCy filtré : {len(common_french_words)} mots")

# -----------------------------
# Fusionner les vocabulaires
# -----------------------------
merged_vocab = sorted(medical_vocab.union(common_french_words))
print(f"Vocabulaire fusionné total : {len(merged_vocab)} mots")

# -----------------------------
# Sauvegarder
# -----------------------------
with open(MERGED_VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(merged_vocab, f, ensure_ascii=False, indent=2)

print(f"Vocabulaire final sauvegardé dans : {MERGED_VOCAB_PATH}")
print("Aperçu (30 premiers mots) :", merged_vocab[:30])
