import os
import csv
import re
import json
from collections import Counter
import spacy
from src.common.config import RESULTS_DIR, VOCAB_DATA_DIR

# -----------------------------
# Chemins
# -----------------------------
CSV_PATH = os.path.join(RESULTS_DIR, "transcriptions.csv")
MEDICAL_PATH = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")
OUTPUT_PATH = os.path.join(VOCAB_DATA_DIR, "optimized_vocab.json")

os.makedirs(VOCAB_DATA_DIR, exist_ok=True)

# -----------------------------
# Chargement du modèle français
# -----------------------------
nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

# Mots français très fréquents à exclure
COMMON_WORDS = {
    "oui", "non", "voila", "bien", "donc", "peut", "etre", "merci", "bonjour",
    "avez", "faire", "fait", "voilà", "mettre", "dire", "aller", "voir", "ok",
    "très", "tout", "comme", "avec", "avoir", "être", "question", "bon", "alors",
    "ben", "peux", "suis", "c'est", "d'accord", "hein", "euh"
}

def clean_word(word: str):
    """Nettoyage de base pour les tokens"""
    w = word.lower().strip()
    if len(w) < 3 or w in stopwords or w in COMMON_WORDS:
        return None
    if re.match(r"^[^a-zA-ZÀ-ÿ'-]+$", w):
        return None
    if any(c.isdigit() for c in w):
        return None
    return w

# -----------------------------
# Étape 1 : Charger le vocabulaire médical
# -----------------------------
with open(MEDICAL_PATH, "r", encoding="utf-8") as f:
    medical_vocab = set(json.load(f))

print(f"🧬 Vocabulaire médical : {len(medical_vocab)} mots chargés")

# -----------------------------
# Étape 2 : Analyser le CSV pour trouver les mots fréquents
# -----------------------------
word_counter = Counter()

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get("transcription_text", "")
        words = re.findall(r"[a-zA-ZÀ-ÿ'-]+", text.lower())
        for w in words:
            cw = clean_word(w)
            if cw:
                word_counter[cw] += 1

# Top 200 mots les plus fréquents dans le corpus
top_common = {w for w, _ in word_counter.most_common(200)}

print(f"📊 Mots fréquents extraits : {len(top_common)}")

# -----------------------------
# Étape 3 : Fusionner intelligemment
# -----------------------------
optimized_vocab = sorted(medical_vocab.union(top_common))

# -----------------------------
# Étape 4 : Sauvegarde
# -----------------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(optimized_vocab, f, ensure_ascii=False, indent=2)

print(f"✅ Vocabulaire optimisé généré : {len(optimized_vocab)} mots")
print(f"💾 Sauvegardé dans : {OUTPUT_PATH}")
print("🧠 Exemple :", optimized_vocab[:30])
