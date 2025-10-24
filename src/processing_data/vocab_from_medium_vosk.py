import os
import json
import ftfy
import unicodedata
from src.common.config import VOCAB_DATA_DIR

vocab_vosk_med = os.path.join(VOCAB_DATA_DIR, "words.txt")

# ------------------ Fonctions utilitaires ------------------

def fix_encoding(token: str) -> str:
    """Répare les encodages bizarres et double-encodage."""
    return ftfy.fix_text(token)

def normalize_ascii(token: str) -> str:
    """Convertit les caractères accentués en leur équivalent ASCII le plus proche."""
    # Corriger ligatures
    token = token.replace('œ', 'oe').replace('Œ', 'Oe')
    # Normalisation NFKD pour séparer lettres et accents
    token = unicodedata.normalize('NFKD', token)
    # Supprimer marques diacritiques
    token = ''.join(c for c in token if unicodedata.category(c) != 'Mn')
    # Convertir en minuscules
    token = token.lower()
    # Garder uniquement ASCII
    token = ''.join(c for c in token if ord(c) < 128)
    return token

def clean_token(token: str) -> str:
    """Nettoyage final pour Vosk : supprime caractères interdits et vides."""
    token = token.strip()
    if not token:
        return None
    # Ignorer caractères invalides pour Vosk
    if any(c in token for c in ['#', '"', '\\', '{', '}', '[', ']', '!']):
        return None
    return token

# ------------------ Lecture et traitement ------------------

tokens = []

with open(vocab_vosk_med, "r", encoding="utf-8") as f:
    for line in f:
        # Supprimer chiffres
        cleaned = ''.join([c for c in line if not c.isdigit()]).strip()
        token = cleaned.split()[0] if cleaned else None
        if token:
            token = fix_encoding(token)
            token = normalize_ascii(token)
            token = clean_token(token)
            if token:
                tokens.append(token)

# ------------------ Déduplication et tri ------------------
tokens = sorted({t for t in tokens if t})

# ------------------ Sauvegarde JSON ------------------
path_json = os.path.join(VOCAB_DATA_DIR, "words_clean.json")
with open(path_json, "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)

print(f"✅ Liste de {len(tokens)} tokens nettoyés enregistrée dans {path_json}")
print(tokens[:50])  # aperçu des 50 premiers tokens
