# # test_tsv.py
# import os
# import pandas as pd
# from jiwer import wer
# from src.common.config import TSV_DIR

# # ---------------------------------------------------------------------
# # Fonction pour charger un TSV et normaliser les colonnes
# # ---------------------------------------------------------------------
# def load_tsv(tsv_path):
#     """
#     Charge un TSV en DataFrame et vérifie que les colonnes attendues existent.
#     """
#     if os.path.exists(tsv_path):
#         df = pd.read_csv(tsv_path, sep="\t")
#         required_cols = ["sentence_id", "sentence", "path"]
#         for col in required_cols:
#             if col not in df.columns:
#                 print(f"[WARN] Colonne {col} manquante dans {tsv_path}")
#                 df[col] = ""
#         return df
#     else:
#         print(f"[WARN] TSV introuvable : {tsv_path}")
#         return pd.DataFrame(columns=["sentence_id", "sentence", "path"])

# # ---------------------------------------------------------------------
# # Fonction pour récupérer le texte de référence pour un fichier audio
# # ---------------------------------------------------------------------
# def get_reference_text(audio_file, validated_df, unvalidated_df):
#     """
#     Retourne le texte de référence pour un fichier audio.
#     Cherche d'abord dans validated, puis dans unvalidated si non trouvé.
#     """
#     # Cherche d'abord dans validated
#     row = validated_df[validated_df['path'] == audio_file]
#     if not row.empty:
#         return str(row.iloc[0]['sentence']).strip()
    
#     # Sinon cherche dans unvalidated
#     row = unvalidated_df[unvalidated_df['path'] == audio_file]
#     if not row.empty:
#         return str(row.iloc[0]['sentence']).strip()
    
#     # Aucun texte trouvé
#     return None

# # ---------------------------------------------------------------------
# # Fonction pour tester le WER
# # ---------------------------------------------------------------------
# def test_wer(audio_file, transcript_generated):
#     """
#     Calcule et affiche le WER pour un fichier audio donné et sa transcription générée.
#     """
#     validated_path = os.path.join(TSV_DIR, "validated.tsv")
#     unvalidated_path = os.path.join(TSV_DIR, "unvalidated.tsv")

#     validated_df = load_tsv(validated_path)
#     unvalidated_df = load_tsv(unvalidated_path)

#     ref_text = get_reference_text(audio_file, validated_df, unvalidated_df)
#     if ref_text is None:
#         print(f"Aucune transcription trouvée pour {audio_file}")
#         return

#     print(f"[DEBUG] ref_text: {ref_text}")
#     print(f"[DEBUG] transcript_generated: {transcript_generated}")

#     # WER sécurisé
#     if len(ref_text.strip()) == 0 or len(transcript_generated.strip()) == 0:
#         print("WER impossible à calculer (texte vide)")
#         return

#     score = wer(ref_text, transcript_generated)
#     print(f"WER pour {audio_file}: {score:.3f}")

# # ---------------------------------------------------------------------
# # Exemple d'utilisation
# # ---------------------------------------------------------------------
# if __name__ == "__main__":
#     # Exemple : nom d'un fichier audio de ton dossier RAW_DATA
#     audio_file_example = "common_voice_fr_41244049.mp3"
#     # Exemple de transcription générée
#     transcript_generated_example = "bonjour ceci est un test"
    
#     test_wer(audio_file_example, transcript_generated_example)


import os
from src.common.config import RAW_DATA_DIR

filename = "common_voice_fr_41499286.mp3"
file_path = os.path.join(RAW_DATA_DIR, filename)

if os.path.exists(file_path):
    print(f"Le fichier {filename} existe dans {RAW_DATA_DIR}")
else:
    print(f"Le fichier {filename} n'existe pas dans {RAW_DATA_DIR}")
