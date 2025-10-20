# import os
# import json
# from src.common.config import PROCESSED_DATA_DIR, TRANSCRIPTS_DIR, VOCAB_DATA_DIR

# # -------------------- Config --------------------
# OUTPUT_JSON = os.path.join(VOCAB_DATA_DIR, "fine_tune_manifest.json")
# os.makedirs(VOCAB_DATA_DIR, exist_ok=True)

# # -------------------- Génération du manifest --------------------
# manifest = []

# audio_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".wav")]
# print(f"{len(audio_files)} fichiers audio trouvés dans {PROCESSED_DATA_DIR}")

# for wav_file in audio_files:
#     base_name = os.path.splitext(wav_file)[0]
#     txt_file = os.path.join(TRANSCRIPTS_DIR, base_name + ".txt")
#     wav_path = os.path.join(PROCESSED_DATA_DIR, wav_file)
    
#     if os.path.exists(txt_file):
#         with open(txt_file, "r", encoding="utf-8") as f:
#             text = f.read().strip()
#         if text:
#             manifest.append({
#                 "audio_filepath": wav_path,
#                 "text": text
#             })
#         else:
#             print(f"[WARNING] Transcription vide pour {wav_file}")
#     else:
#         print(f"[WARNING] Transcription manquante pour {wav_file}")

# # -------------------- Sauvegarde JSON --------------------
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(manifest, f, ensure_ascii=False, indent=2)

# print(f"Manifest JSON créé : {OUTPUT_JSON}")
# print(f"Nombre de fichiers prêts pour fine-tuning : {len(manifest)}")
