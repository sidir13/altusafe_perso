import os
import json
import spacy
import re
import csv
from collections import Counter
from src.common.config import RESULTS_DIR, VOCAB_DATA_DIR

# -----------------------------
# Chemins
# -----------------------------
CSV_PATH = os.path.join(RESULTS_DIR, "transcriptions.csv")
VOCAB_PATH = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")

# -----------------------------
# Chargement du vocabulaire initial
# -----------------------------
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_list = json.load(f)

print(f"🔍 Vocabulaire initial : {len(vocab_list)} mots")

# -----------------------------
# Préparation des filtres
# -----------------------------
nlp = spacy.load("fr_core_news_sm")
stopwords = nlp.Defaults.stop_words

COMMON_WORDS = {
    "amene", "savez", "donc", "bien", "peut", "etre", "voila", "avez", "fait", "faire",
    "aller", "dire", "voir", "mettre", "venir", "vouloir", "savoir", "pouvoir", "donner",
    "prendre", "trouver", "passer", "falloir", "devoir", "regarder", "demander", "bonjour",
    "merci", "d'accord", "oui", "non", "voilà", "ben", "ok", "question", "attention",
    "mettrez", "très", "tout", "comme", "avec", "avoir", "être", "faire", "dire", "voir", "aller"
}

def is_noisy(word: str) -> bool:
    return (
        len(word) < 3
        or any(c.isdigit() for c in word)
        or re.match(r"^[^a-zA-ZÀ-ÿ'-]+$", word)
        or word.startswith(("d'", "l'", "qu'", "j'", "n'", "s'"))
        or word in stopwords
        or word in COMMON_WORDS
    )

# -----------------------------
# Étape 1 : filtrer le vocabulaire
# -----------------------------
filtered_vocab = [w for w in vocab_list if not is_noisy(w)]
print(f"✅ Après filtrage : {len(filtered_vocab)} mots conservés")

# -----------------------------
# Étape 2 : compter les fréquences dans le corpus
# -----------------------------
word_freq = Counter()

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row.get("transcription_text", "")
        words = re.findall(r"[a-zA-ZÀ-ÿ'-]+", text.lower())
        for w in words:
            if w in filtered_vocab:
                word_freq[w] += 1

# -----------------------------
# Étape 3 : garder les 100 plus fréquents
# -----------------------------
top_100 = [w for w, _ in word_freq.most_common(100)]

# -----------------------------
# Étape 4 : sauvegarder dans le même fichier
# -----------------------------
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(sorted(top_100), f, ensure_ascii=False, indent=2)

print(f"💾 Nouveau vocabulaire sauvegardé dans {VOCAB_PATH} (100 mots)")
print("🏆 Exemple de mots :", sorted(top_100)[:20])
