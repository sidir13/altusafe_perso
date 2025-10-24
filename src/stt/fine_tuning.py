import os
import json
import subprocess
from src.common.config import DEFAULT_MODEL_FR, PROCESSED_DIR, TRANSCRIPTS_DIR, VOCAB_DATA_DIR

# -------------------- Chemins --------------------
manifest_path = os.path.join(VOCAB_DATA_DIR, "fine_tune_manifest.json")
output_model_dir = os.path.join(VOCAB_DATA_DIR, "vosk-model-finetuned-medical")
os.makedirs(output_model_dir, exist_ok=True)

# -------------------- Génération du manifest --------------------
audio_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith(".wav")])
manifest = {"audio": [], "text": []}

for audio_file in audio_files:
    audio_path = os.path.join(PROCESSED_DIR, audio_file)
    txt_file = os.path.splitext(audio_file)[0] + ".txt"
    txt_path = os.path.join(TRANSCRIPTS_DIR, txt_file)

    if not os.path.exists(txt_path):
        print(f"[WARNING] Transcription introuvable pour {audio_file}, fichier ignoré")
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()
        if transcript == "":
            print(f"[WARNING] Transcription vide pour {audio_file}, fichier ignoré")
            continue

    manifest["audio"].append(audio_path)
    manifest["text"].append(transcript)

# Sauvegarde du manifest
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Manifest JSON créé : {manifest_path}")
print(f"Nombre de fichiers prêts pour fine-tuning : {len(manifest['audio'])}")

# -------------------- Lancement du fine-tuning --------------------
train_command = [
    "vosk-train",               
    DEFAULT_MODEL_FR,
    manifest_path,
    output_model_dir
]

print("Commande fine-tuning :", " ".join(train_command))
subprocess.run(train_command)
print(f"Fine-tuning terminé. Modèle sauvegardé dans : {output_model_dir}")
