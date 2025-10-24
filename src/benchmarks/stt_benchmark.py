import os
import time
import json
import argparse
import psutil
import wave
import csv
import subprocess
import logging
import pandas as pd
from vosk import Model, KaldiRecognizer
from jiwer import wer
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import sacrebleu
import spacy  

from src.common.config import (
    DEFAULT_MODEL_FR,
    RAW_DATA_DIR,
    TRANSCRIPTS_DIR,
    RESULTS_DIR,
    TSV_DIR
)

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
logger = logging.getLogger("STT_Benchmark")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# ---------------------------------------------------------------------
# Charger le modèle de lemmatisation
# ---------------------------------------------------------------------
try:
    nlp = spacy.load("fr_core_news_md")
except OSError:
    nlp = spacy.load("fr_core_news_sm")

def lemmatize_text(text):
    """Renvoie une version lemmatisée (canonique) du texte en français."""
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmas)

# ---------------------------------------------------------------------
# Charger les TSV
# ---------------------------------------------------------------------
def load_tsvs():
    validated_path = os.path.join(TSV_DIR, "validated.tsv")
    invalidated_path = os.path.join(TSV_DIR, "invalidated.tsv")
    clips_duration_path = os.path.join(TSV_DIR, "clip_durations.tsv")

    validated_df = pd.read_csv(validated_path, sep="\t") if os.path.exists(validated_path) else pd.DataFrame(columns=["path", "sentence"])
    invalidated_df = pd.read_csv(invalidated_path, sep="\t") if os.path.exists(invalidated_path) else pd.DataFrame(columns=["path", "sentence"])
    clips_duration_df = pd.read_csv(clips_duration_path, sep="\t") if os.path.exists(clips_duration_path) else pd.DataFrame(columns=["clip", "duration[ms]"])

    return validated_df, invalidated_df, clips_duration_df

validated_df, invalidated_df, clips_duration_df = load_tsvs()

# ---------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------
def convert_to_wav(input_path, temp_filename="temp.wav"):
    temp_path = os.path.join(os.path.dirname(input_path), temp_filename)
    subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", temp_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
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
    mem = process.memory_info().rss / (1024 * 1024)
    return round(mem, 2)

def load_reference_text(audio_file):
    base_name = os.path.basename(audio_file)
    for df in [validated_df, invalidated_df]:
        row = df[df['path'].apply(lambda x: os.path.basename(x)) == base_name]
        if not row.empty:
            return str(row.iloc[0]['sentence']).strip()
    txt_path = os.path.join(TRANSCRIPTS_DIR, base_name.replace(".mp3", ".txt"))
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

def get_clip_duration(audio_file):
    base_name = os.path.basename(audio_file)
    row = clips_duration_df[clips_duration_df["clip"] == base_name]
    if not row.empty:
        return row["duration[ms]"].values[0] / 1000
    return None

def write_csv(result, output_csv):
    header = [
        "audio_file", "model", "latency_sec", "memory_mb", "wer",
        "wer_token", "levenshtein", "levenshtein_pct", "accuracy",
        "bleu3", "meteor", "chrf", "rougeL",
        "reference_text", "reference_text_lemma",
        "transcript", "transcript_lemma",
        "duration_sec", "latency_per_sec", "memory_per_sec",
        "tokens", "tokens_per_sec"
    ]
    file_exists = os.path.exists(output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark Vosk STT")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_FR)
    parser.add_argument("--audio_dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_dir.rstrip("/\\"))
    results_path = os.path.join(args.results_dir, f"benchmark_{model_name}.csv")

    audio_files = [f for f in os.listdir(args.audio_dir)
                   if f.lower().endswith((".wav", ".mp3", ".mp4", ".flac", ".m4a", ".ogg"))]

    if not audio_files:
        logger.warning(f"Aucun fichier audio/vidéo trouvé dans {args.audio_dir}")
        return

    logger.info(f"Benchmark du modèle : {args.model_dir}")
    logger.info(f"Nombre d'audios testés : {len(audio_files)} fichiers")

    model = Model(args.model_dir)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for audio_file in audio_files:
        input_path = os.path.join(args.audio_dir, audio_file)
        logger.info(f"Traitement de {audio_file} ...")

        mem_before = measure_memory()
        transcript, latency = transcribe_audio(model, input_path)
        mem_after = measure_memory()

        ref_text = load_reference_text(audio_file)
        duration_sec = get_clip_duration(audio_file)
        num_tokens = len(transcript.split())

        # Initialiser toutes les valeurs pour éviter les erreurs
        wer_score = wer_token_score = levenshtein_score = levenshtein_pct = acc_score = bleu_score = meteor = chrf_score = rouge_l_score = "N/A"
        ref_text_lemma = transcript_lemma = "N/A"

        if ref_text:
            try:
                # ✅ Lemmatisation
                ref_text_lemma = lemmatize_text(ref_text)
                transcript_lemma = lemmatize_text(transcript)

                # Character-level metrics (lemmatisés)
                wer_score = wer(ref_text_lemma, transcript_lemma)
                levenshtein_score = Levenshtein.distance(ref_text_lemma, transcript_lemma)
                levenshtein_pct = levenshtein_score / max(len(ref_text_lemma), 1)

                # Token-level
                ref_words = ref_text_lemma.split()
                hyp_words = transcript_lemma.split()
                correct_words = sum(r == h for r, h in zip(ref_words, hyp_words))
                wer_token_score = 1 - sum(r != h for r, h in zip(ref_words, hyp_words)) / max(len(ref_words), 1)
                acc_score = correct_words / max(len(ref_words), 1)

                # BLEU3
                bleu_score = sentence_bleu([ref_words], hyp_words, weights=(1/3, 1/3, 1/3, 0))

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
            "latency_sec": round(latency, 3),
            "memory_mb": round(mem_after - mem_before, 2),
            "wer": round(wer_score, 3) if isinstance(wer_score, float) else wer_score,
            "wer_token": round(wer_token_score, 3) if isinstance(wer_token_score, float) else wer_token_score,
            "levenshtein": round(levenshtein_score, 3) if isinstance(levenshtein_score, float) else levenshtein_score,
            "levenshtein_pct": round(levenshtein_pct, 3) if isinstance(levenshtein_pct, float) else levenshtein_pct,
            "accuracy": round(acc_score, 3) if isinstance(acc_score, float) else acc_score,
            "bleu3": round(bleu_score, 3) if isinstance(bleu_score, float) else bleu_score,
            "meteor": round(meteor, 3) if isinstance(meteor, float) else meteor,
            "chrf": round(chrf_score, 3) if isinstance(chrf_score, float) else chrf_score,
            "rougeL": round(rouge_l_score, 3) if isinstance(rouge_l_score, float) else rouge_l_score,
            "reference_text": ref_text if ref_text else "N/A",
            "reference_text_lemma": ref_text_lemma if ref_text else "N/A",
            "transcript": transcript,
            "transcript_lemma": transcript_lemma,
            "duration_sec": round(duration_sec, 3) if duration_sec else "N/A",
            "latency_per_sec": round(latency / duration_sec, 3) if duration_sec else "N/A",
            "memory_per_sec": round((mem_after - mem_before) / duration_sec, 3) if duration_sec else "N/A",
            "tokens": num_tokens,
            "tokens_per_sec": round(num_tokens / duration_sec, 3) if duration_sec else "N/A"
        }

        write_csv(result, results_path)
        logger.info(f"{audio_file} traité : Lat {latency:.2f}s, Mem {mem_after - mem_before:.1f} Mo, "
                    f"WER={result['wer']}, Token-WER={result['wer_token']}, Levenshtein={result['levenshtein']}, "
                    f"BLEU3={result['bleu3']}, METEOR={result['meteor']}, chrF={result['chrf']}, ROUGE-L={result['rougeL']}")

if __name__ == "__main__":
    main()
