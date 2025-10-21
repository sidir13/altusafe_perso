# Inference Vosk Mini – Transcription Audio

Ce package permet de transcrire des fichiers audio en français en utilisant le modèle **Vosk mini**. 
Il génère à la fois un fichier texte de transcription et un CSV récapitulatif.

---

## Prérequis

- Python **3.12.2** installé sur la machine.  
- Un environnement virtuel recommandé pour isoler les dépendances.

---

## Contenu du package

- `src/inference/run_stt_csv_vosk.py`  
  Script Python principal pour effectuer la transcription d’un fichier audio.

- `MODELS_DIR/vosk-model-small-fr/`  
  Modèle Vosk mini français (doit être présent sur la machine).

- `INFERENCE_DIR/`  
  Dossier où sera créé le fichier CSV `transcription_inference_vosk.csv`.

- `TRANSCRIPTS_DIR/`  
  Dossier où seront stockées les transcriptions `.txt` individuelles.

- `requirements.txt`  
  Dépendances Python nécessaires.

---

## Installation

1. Créer un environnement virtuel :

```powershell
# Windows PowerShell
python -m venv venv_inference
venv_inference\Scripts\Activate.ps1


pip install -r src/inference/requirements.txt
(vosk==0.3.45)


2. Vérifier que le modèle Vosk mini français est présent dans :
models/vosk-model-small-fr-0.22


## Utilisation

Pour transcrire un fichier audio unique :

python -m src.inference.run_stt_csv_vosk <chemin/vers/fichier.wav>

Exemple : 

python -m src.inference.run_stt_csv_vosk data/processed/wav_data_v2/enreJerome1_seg0.wav
