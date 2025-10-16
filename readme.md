# Reconnaissance vocale (STT) – Projet Altusafe

## Objectif

Optimiser l'intégration de Vosk (ou autre modèle STT) pour un fonctionnement fluide sur tablette Android (12Go RAM) avec les contraintes suivantes :

- Fonctionnement 100% offline
- Latences minimales pour un assistant réactif
- Environnement bruité (bloc opératoire)
- Support bilingue français/anglais
- Détection possible si le micro est partiellement obstrué par une coque

---

## Approche

- Évaluer différents modèles STT (Vosk et alternatives) pour Android
- Optimiser pour réduire latences et consommation mémoire
- Tester sur tablette Samsung S10+ (device de référence)
- Implémenter une détection de micro obstrué pour alerter l’utilisateur si nécessaire

---

## Données

- Enregistrements réels de blocs opératoires fournis par Altusafe (sous NDA)
- Possibilité de générer des données synthétiques si nécessaire pour le développement ou les tests

---

## Livrables attendus

- Modèle Vosk optimisé et intégré
- Documentation de performance et recommandations
- Scripts de test et pipeline de benchmark

---

## Scripts et état actuel

| Script | Description | État / Commentaire |
|--------|-------------|------------------|
| `src/benchmarks/stt_benchmark.py` | Benchmark Vosk STT sur fichiers audio/vidéo | ✅ Implémenté : mesure **latence**, **mémoire**, **WER**. Premier outil pour évaluer différents modèles. |
| `src/common/config.py` | Centralise tous les chemins de fichiers et dossiers | ✅ Implémenté : permet une maintenance facile et la réutilisation des chemins dans tous les scripts. |
| `src/android/integration.py` | Script prévu pour intégration STT sur tablette Android | ⏳ À venir |
| `src/micro_detection/mic_obstruction.py` | Script prévu pour détecter si le micro est obstrué | ⏳ À venir |
| `src/data/synthetic_generation.py` | Script prévu pour générer des fichiers audio synthétiques | ⏳ À venir |

> Les scripts marqués “À venir” seront ajoutés au fur et à mesure que le projet avance.

---

## Notes techniques

Tous les fichiers audio sont convertis automatiquement en WAV mono 16kHz pour compatibilité avec Vosk.

Version de Python utilisée : 3.13.9

Projet évolutif : chaque nouvelle fonctionnalité (intégration Android, détection micro, génération de données) aura sa section et son script dédié.