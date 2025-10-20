import os
import json
import wave
import logging
import time
import psutil
import csv
from vosk import Model, KaldiRecognizer
from jiwer import wer
import spacy
from src.common.config import DEFAULT_MODEL_FR, PROCESSED_DATA_DIR, TRANSCRIPTS_DIR, RESULTS_DIR, VOCAB_DATA_DIR

# -------------------- Logger --------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -------------------- Lemmatisation --------------------
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    nlp = spacy.load("fr_core_news_sm")

def lemmatize_text(text):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_punct and not t.is_space])

# -------------------- Chargement du modèle Vosk --------------------
logger.info(f"Chargement du modèle Vosk : {DEFAULT_MODEL_FR}")
model = Model(DEFAULT_MODEL_FR)

# -------------------- Charger vocabulaire --------------------
vocab_path = os.path.join(VOCAB_DATA_DIR, "medical_vocab.json")
vocab = []
if os.path.exists(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    logger.info(f"Vocabulaire chargé ({len(vocab)} mots)")
else:
    logger.warning(f"Vocabulaire introuvable : {vocab_path}")

# -------------------- Préparer CSV --------------------
CSV_PATH = os.path.join(RESULTS_DIR, "stt_benchmark_medecin_with_vocab.csv")
csv_exists = os.path.exists(CSV_PATH)
csv_file = open(CSV_PATH, mode="a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=[
    "audio_file", "latency_sec", "memory_mb", "wer", "accuracy",
    "transcript", "transcript_lemma", "reference_lemma"
])
if not csv_exists:
    writer.writeheader()

# -------------------- Fonctions utilitaires --------------------
def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024*1024)

def transcribe_audio(audio_path, vocab=None):
    wf = wave.open(audio_path, "rb")
    # Injection du vocabulaire JSON dans le recognizer
    rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(vocab) if vocab else None)
    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += " " + res.get("text", "")
    final_res = json.loads(rec.FinalResult())
    result_text += " " + final_res.get("text", "")
    wf.close()
    return result_text.strip()

# -------------------- Traitement des fichiers --------------------
audio_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith(".wav")]
logger.info(f"{len(audio_files)} fichiers audio trouvés dans {PROCESSED_DATA_DIR}")

for audio_file in audio_files:
    audio_path = os.path.join(PROCESSED_DATA_DIR, audio_file)
    logger.info(f"Transcription de : {audio_file}")

    mem_before = measure_memory()
    start_time = time.time()
    transcript = transcribe_audio(audio_path, vocab)
    latency = time.time() - start_time
    mem_after = measure_memory()

    transcript_lemma = lemmatize_text(transcript)

    # Charger texte de référence
    ref_path = os.path.join(TRANSCRIPTS_DIR, os.path.splitext(audio_file)[0] + ".txt")
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        ref_lemma = lemmatize_text(ref_text)

        # Calcul métriques
        wer_score = wer(ref_lemma, transcript_lemma)
        ref_tokens = ref_lemma.split()
        hyp_tokens = transcript_lemma.split()
        correct_tokens = sum(r==h for r,h in zip(ref_tokens, hyp_tokens))
        accuracy = correct_tokens / max(len(ref_tokens),1)
    else:
        logger.warning(f"Texte de référence introuvable pour {audio_file}")
        ref_lemma = ""
        wer_score = None
        accuracy = None

    # Écrire dans CSV
    writer.writerow({
        "audio_file": audio_file,
        "latency_sec": round(latency,3),
        "memory_mb": round(mem_after - mem_before,3),
        "wer": round(wer_score,3) if wer_score is not None else "N/A",
        "accuracy": round(accuracy,3) if accuracy is not None else "N/A",
        "transcript": transcript,
        "transcript_lemma": transcript_lemma,
        "reference_lemma": ref_lemma
    })
    logger.info(f"{audio_file} traité : Latency={latency:.2f}s, Memory={mem_after-mem_before:.2f}Mo, WER={wer_score}, Accuracy={accuracy}")

csv_file.close()
logger.info(f"CSV des résultats enregistré dans : {CSV_PATH}")
