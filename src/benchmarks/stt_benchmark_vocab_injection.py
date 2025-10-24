import os
import time
import json
import csv
import logging
import wave
import subprocess
import psutil
from tqdm import tqdm
from vosk import Model, KaldiRecognizer
from jiwer import wer
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
import spacy

# ----------------------- Configuration -----------------------
from src.common.config import (
    WAV_DATA_DIR,
    TRANSCRIPTS_DIR,
    RESULTS_DIR,
    EXPERIMENTAL_MODEL_FR,
    SAMPLE_RATE,
    VOCAB_DATA_DIR
)

# ----------------------- Logger -----------------------
LOG_PATH = os.path.join(RESULTS_DIR, "benchmark_medical.log")
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------- Lemmatisation -----------------------
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    nlp = spacy.load("fr_core_news_sm")

def lemmatize_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

# ----------------------- Charger et fusionner le vocabulaire -----------------------
VOCAB_JSON_PATH = os.path.join(VOCAB_DATA_DIR, "words_clean.json")
if os.path.exists(VOCAB_JSON_PATH):
    with open(VOCAB_JSON_PATH, "r", encoding="utf-8") as f:
        generated_vocab = json.load(f)
else:
    generated_vocab = []

OPTIMIZED_VOCAB_PATH = os.path.join(VOCAB_DATA_DIR, "optimized_vocab.json")
with open(OPTIMIZED_VOCAB_PATH, "r", encoding="utf-8") as f:
    MEDICAL_VOCABULARY = json.load(f)

# Fusionner vocabulaire manuel et généré
# FULL_VOCABULARY = sorted(set(MEDICAL_VOCABULARY + generated_vocab))
FULL_VOCABULARY = sorted(set(generated_vocab))
logger.info(f"Taille du vocabulaire injecté : {len(FULL_VOCABULARY)} mots")

# ----------------------- Fonctions utilitaires -----------------------
def convert_to_wav(input_path, temp_filename="temp.wav"):
    temp_path = os.path.join(os.path.dirname(input_path), temp_filename)
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", temp_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return temp_path

def transcribe_with_vocab(model, audio_path):
    wav_path = convert_to_wav(audio_path)
    result_text = ""
    start_time = time.time()
    try:
        with wave.open(wav_path, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate(), json.dumps(FULL_VOCABULARY))
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    result_text += " " + res.get("text", "")
            final_res = json.loads(rec.FinalResult())
            result_text += " " + final_res.get("text", "")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    latency = time.time() - start_time
    return result_text.strip(), latency

def measure_memory():
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024*1024), 2)

def load_reference_text(audio_file):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    reference_path = os.path.join(TRANSCRIPTS_DIR, f"{base_name}.txt")
    if os.path.exists(reference_path):
        with open(reference_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

def write_csv(result, output_csv):
    header = [
        "audio_file", "model", "latency_sec", "memory_mb", "wer",
        "wer_token", "levenshtein", "levenshtein_pct", "accuracy",
        "bleu3", "meteor", "chrf", "rougeL",
        "reference_text", "reference_text_lemma",
        "transcript", "transcript_lemma",
        "tokens"
    ]
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

# ----------------------- Script principal -----------------------
def main():
    model_name = os.path.basename(EXPERIMENTAL_MODEL_FR.rstrip("/\\"))
    results_path = os.path.join(RESULTS_DIR, f"benchmark_medical_v4.csv")

    audio_files = [f for f in os.listdir(WAV_DATA_DIR) if f.lower().endswith(".wav")]
    if not audio_files:
        logger.warning(f"Aucun fichier audio trouvé dans {WAV_DATA_DIR}")
        return

    logger.info(f"Benchmark du modèle médical : {EXPERIMENTAL_MODEL_FR}")
    logger.info(f"Nombre d'audios : {len(audio_files)} fichiers")

    model = Model(EXPERIMENTAL_MODEL_FR)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for audio_file in tqdm(audio_files, desc="Benchmark", unit="fichier"):
        input_path = os.path.join(WAV_DATA_DIR, audio_file)
        mem_before = measure_memory()
        transcript, latency = transcribe_with_vocab(model, input_path)
        mem_after = measure_memory()

        ref_text = load_reference_text(audio_file)
        num_tokens = len(transcript.split())

        # Initialisation
        wer_score = wer_token_score = levenshtein_score = levenshtein_pct = acc_score = bleu_score = meteor = chrf_score = rouge_l_score = 0.0
        ref_text_lemma = transcript_lemma = ""

        if ref_text and transcript:
            try:
                ref_text_lemma = lemmatize_text(ref_text)
                transcript_lemma = lemmatize_text(transcript)

                # Character-level
                wer_score = wer(ref_text_lemma, transcript_lemma)
                levenshtein_score = Levenshtein.distance(ref_text_lemma, transcript_lemma)
                levenshtein_pct = levenshtein_score / max(len(ref_text_lemma),1)

                # Token-level
                ref_words = ref_text_lemma.split()
                hyp_words = transcript_lemma.split()
                correct_words = sum(r==h for r,h in zip(ref_words,hyp_words))
                wer_token_score = 1 - sum(r!=h for r,h in zip(ref_words,hyp_words))/max(len(ref_words),1)
                acc_score = correct_words / max(len(ref_words),1)

                # BLEU3
                bleu_score = sentence_bleu([ref_words], hyp_words, weights=(1/3,1/3,1/3,0))

                # METEOR
                meteor = meteor_score([ref_text_lemma], transcript_lemma)

                # chrF
                chrf_score = sacrebleu.corpus_chrf([transcript_lemma], [[ref_text_lemma]])

                # ROUGE-L
                rouge_l_score = scorer.score(ref_text_lemma, transcript_lemma)['rougeL'].fmeasure

            except Exception as e:
                logger.warning(f"Erreur métriques pour {audio_file}: {e}")

        result = {
            "audio_file": audio_file,
            "model": model_name,
            "latency_sec": round(latency,3),
            "memory_mb": round(mem_after-mem_before,2),
            "wer": round(wer_score,3),
            "wer_token": round(wer_token_score,3),
            "levenshtein": round(levenshtein_score,3),
            "levenshtein_pct": round(levenshtein_pct,3),
            "accuracy": round(acc_score,3),
            "bleu3": round(bleu_score,3),
            "meteor": round(meteor,3),
            "chrf": round(chrf_score,3),
            "rougeL": round(rouge_l_score,3),
            "reference_text": ref_text if ref_text else "N/A",
            "reference_text_lemma": ref_text_lemma if ref_text else "N/A",
            "transcript": transcript,
            "transcript_lemma": transcript_lemma,
            "tokens": num_tokens
        }

        write_csv(result, results_path)

    logger.info(f"Toutes les métriques ont été enregistrées dans : {results_path}")

if __name__ == "__main__":
    main()
