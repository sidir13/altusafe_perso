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
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4') 

from src.common.config import WAV_DATA_DIR_v2, TRANSCRIPTS_DIR, RESULTS_DIR, DEFAULT_MODEL_FR

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
LOG_PATH = os.path.join(RESULTS_DIR, "benchmark_v2.log")
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

# ---------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    nlp = spacy.load("fr_core_news_sm")

def lemmatize_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space])

# ---------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------
def convert_to_wav(input_path, temp_filename="temp.wav"):
    temp_path = os.path.join(os.path.dirname(input_path), temp_filename)
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", temp_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    return temp_path

def transcribe_audio(model, input_path):
    wav_path = convert_to_wav(input_path)
    result_text = ""
    start_time = time.time()
    try:
        with wave.open(wav_path, "rb") as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
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

# ---------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------
def main():
    model_name = os.path.basename(DEFAULT_MODEL_FR.rstrip("/\\"))
    results_path = os.path.join(RESULTS_DIR, f"benchmark_{model_name}_v2.csv")

    audio_files = [f for f in os.listdir(WAV_DATA_DIR_v2) if f.lower().endswith(".wav")]
    if not audio_files:
        logger.warning(f"Aucun fichier audio trouvé dans {WAV_DATA_DIR_v2}")
        return

    logger.info(f"Benchmark du modèle : {DEFAULT_MODEL_FR}")
    logger.info(f"Nombre d'audios : {len(audio_files)} fichiers")

    model = Model(DEFAULT_MODEL_FR)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for audio_file in tqdm(audio_files, desc="Benchmark", unit="fichier"):
        input_path = os.path.join(WAV_DATA_DIR_v2, audio_file)
        mem_before = measure_memory()
        transcript, latency = transcribe_audio(model, input_path)
        mem_after = measure_memory()

        ref_text = load_reference_text(audio_file)
        num_tokens = len(transcript.split()) if transcript else 0

        # Initialisation
        if ref_text and transcript:
            ref_text_lemma = lemmatize_text(ref_text)
            transcript_lemma = lemmatize_text(transcript)
            ref_words = ref_text_lemma.split()
            hyp_words = transcript_lemma.split()
            try:
                # Character-level
                wer_score = wer(ref_text_lemma, transcript_lemma)
                levenshtein_score = Levenshtein.distance(ref_text_lemma, transcript_lemma)
                levenshtein_pct = levenshtein_score / max(len(ref_text_lemma), 1)

                # Token-level
                correct_words = sum(r==h for r,h in zip(ref_words,hyp_words))
                wer_token_score = 1 - sum(r!=h for r,h in zip(ref_words,hyp_words))/max(len(ref_words),1)
                acc_score = correct_words / max(len(ref_words),1)

                # BLEU3
                try:
                    bleu_score = sentence_bleu([ref_words], hyp_words, weights=(1/3,1/3,1/3,0))
                except Exception as e:
                    logger.warning(f"BLEU3 error {audio_file}: {e}")
                    bleu_score = "N/A"

                # METEOR
                try:
                    meteor = meteor_score([ref_words], hyp_words)
                except Exception as e:
                    logger.warning(f"METEOR error {audio_file}: {e}")
                    meteor = "N/A"

                # chrF
                try:
                    chrf_obj = sacrebleu.corpus_chrf([transcript_lemma], [[ref_text_lemma]])
                    chrf_score = chrf_obj.score
                except Exception as e:
                    logger.warning(f"chrF error {audio_file}: {e}")
                    chrf_score = "N/A"

                # ROUGE-L
                try:
                    rouge_l_score = scorer.score(ref_text_lemma, transcript_lemma)['rougeL'].fmeasure
                except Exception as e:
                    logger.warning(f"ROUGE-L error {audio_file}: {e}")
                    rouge_l_score = "N/A"

            except Exception as e:
                logger.warning(f"Métriques échouées pour {audio_file}: {e}")
                wer_score = wer_token_score = levenshtein_score = levenshtein_pct = acc_score = bleu_score = meteor = chrf_score = rouge_l_score = "N/A"

        else:
            ref_text_lemma = transcript_lemma = ""
            wer_score = wer_token_score = levenshtein_score = levenshtein_pct = acc_score = bleu_score = meteor = chrf_score = rouge_l_score = "N/A"

        result = {
            "audio_file": audio_file,
            "model": model_name,
            "latency_sec": round(latency,3),
            "memory_mb": round(mem_after-mem_before,2),
            "wer": round(wer_score,3) if isinstance(wer_score,float) else wer_score,
            "wer_token": round(wer_token_score,3) if isinstance(wer_token_score,float) else wer_token_score,
            "levenshtein": round(levenshtein_score,3) if isinstance(levenshtein_score,float) else levenshtein_score,
            "levenshtein_pct": round(levenshtein_pct,3) if isinstance(levenshtein_pct,float) else levenshtein_pct,
            "accuracy": round(acc_score,3) if isinstance(acc_score,float) else acc_score,
            "bleu3": round(bleu_score,3) if isinstance(bleu_score,float) else bleu_score,
            "meteor": round(meteor,3) if isinstance(meteor,float) else meteor,
            "chrf": round(chrf_score,3) if isinstance(chrf_score,float) else chrf_score,
            "rougeL": round(rouge_l_score,3) if isinstance(rouge_l_score,float) else rouge_l_score,
            "reference_text": ref_text if ref_text else "N/A",
            "reference_text_lemma": ref_text_lemma if ref_text else "N/A",
            "transcript": transcript if transcript else "N/A",
            "transcript_lemma": transcript_lemma if transcript else "N/A",
            "tokens": num_tokens
        }

        write_csv(result, results_path)

    logger.info(f"📁 Toutes les métriques v2 ont été enregistrées dans : {results_path}")


if __name__ == "__main__":
    main()
