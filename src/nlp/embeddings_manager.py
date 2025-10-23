# src/nlp/embeddings_manager.py
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsManager:
    def __init__(self, vocab_path, model_name="camembert-base", cache_dir=None):
        """
        vocab_path : chemin vers le fichier JSON contenant le vocabulaire
        model_name : modèle Hugging Face à utiliser (ici français biomédical)
        cache_dir : chemin pour sauvegarder le cache des embeddings
        """
        self.vocab_path = vocab_path
        self.model_name = model_name
        self.cache_path = cache_dir or os.path.splitext(vocab_path)[0] + "_embeddings.pt"

        # Charger vocabulaire
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = list(json.load(f))

        # Charger modèle Transformers français
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Charger ou générer embeddings
        self.embeddings = self._load_or_build_embeddings()

    def _get_embedding(self, text):
        """Retourne l'embedding vectoriel moyen d'un texte/mot"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def _load_or_build_embeddings(self):
        """Charge le cache si présent ou calcule les embeddings"""
        if os.path.exists(self.cache_path):
            print(f"⚡ Chargement des embeddings depuis le cache : {self.cache_path}")
            import numpy
            # ⚠️ Permet de loader les objets picklés numpy avec PyTorch 2.6+
            with torch.serialization.safe_globals([numpy._core.multiarray._reconstruct]):
                return torch.load(self.cache_path, weights_only=False)

        print("🧬 Calcul des embeddings du vocabulaire...")
        embeddings = {}
        for word in self.vocab:
            embeddings[word] = self._get_embedding(word)
        torch.save(embeddings, self.cache_path)
        print(f"✅ Embeddings sauvegardés dans : {self.cache_path}")
        return embeddings

    def find_best_match(self, word):
        """Trouve le mot du vocabulaire le plus proche selon la similarité cosine"""
        try:
            word_emb = self._get_embedding(word)
        except Exception:
            return word, 0.0

        best_word, best_score = word, 0.0
        for vocab_word, vocab_emb in self.embeddings.items():
            score = cosine_similarity(word_emb, vocab_emb)[0][0]
            if score > best_score:
                best_score = score
                best_word = vocab_word
        return best_word, best_score
