# src/nlp/medical_postprocessor.py
import json
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings_manager import EmbeddingsManager

class MedicalPostProcessorPhonetic:
    def __init__(self, vocab_json_path, threshold=0.5, top_n=5):
        """
        Post-traitement phonétique + sémantique avec vocabulaire pré-calculé.
        vocab_json_path : chemin vers le JSON phonétique {mot: phonétique}
        """
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            self.vocab_phon = json.load(f)

        self.emb_manager = EmbeddingsManager(vocab_json_path)
        self.threshold = threshold
        self.top_n = top_n

    def _phonetic_distance(self, word1, word2):
        """Distance Levenshtein normalisée entre deux représentations phonétiques"""
        phon1 = self.vocab_phon.get(word1, word1)
        phon2 = self.vocab_phon.get(word2, word2)
        dist = Levenshtein.distance(phon1, phon2)
        max_len = max(len(phon1), len(phon2), 1)
        return dist / max_len

    def process_sentence(self, sentence: str):
        """
        Corrige une phrase selon la similarité phonétique et sémantique contextuelle.
        """
        words = sentence.split()
        corrected_words = []
        replacements = {}
        cosine_scores = {}

        # Embedding de la phrase originale
        phrase_emb_original = self.emb_manager._get_embedding(sentence)

        for i, word in enumerate(words):
            # Sélection des N mots phonétiquement les plus proches
            candidates = sorted(
                self.vocab_phon.keys(),
                key=lambda w: self._phonetic_distance(word, w)
            )[:self.top_n]

            best_word = word
            best_score = -1.0

            #  Test contextuel : remplace le mot dans la phrase et compare l'embedding global
            for candidate in candidates:
                test_phrase = " ".join(
                    words[:i] + [candidate] + words[i + 1 :]
                )
                phrase_emb_candidate = self.emb_manager._get_embedding(test_phrase)
                score = cosine_similarity(
                    phrase_emb_candidate, phrase_emb_original
                )[0][0]

                if score > best_score:
                    best_score = score
                    best_word = candidate

            # Appliquer le remplacement si le score dépasse le seuil
            if best_score >= self.threshold:
                corrected_words.append(best_word)
                if best_word != word:
                    replacements[word] = best_word
                    cosine_scores[word] = float(best_score)  
            else:
                corrected_words.append(word)

        corrected_sentence = " ".join(corrected_words)
        return corrected_sentence, replacements, cosine_scores
