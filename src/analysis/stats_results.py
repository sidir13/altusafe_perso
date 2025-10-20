import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np
from src.common.config import REPORTING_DIR, RESULTS_DIR

# ---------------------------------------------------------------------
# Lecture des résultats
# ---------------------------------------------------------------------
BENCHMARK_CSV = os.path.join(RESULTS_DIR, "medecin/benchmark_vosk-model-small-fr-0.22_medecin.csv")
df = pd.read_csv(BENCHMARK_CSV)

# Nettoyage : mémoire négative → 0
df['memory_mb'] = df['memory_mb'].apply(lambda x: max(0, x))

# Colonnes numériques à convertir
num_cols = ['wer','wer_token','levenshtein','levenshtein_pct','accuracy','bleu3',
            'meteor','chrf','rougeL','latency_sec','memory_mb','duration_sec',
            'latency_per_sec','memory_per_sec','tokens','tokens_per_sec',
            'latency_per_token','memory_per_token','wer_per_token']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['transcript']).copy()

# Créer dossier reporting
os.makedirs(REPORTING_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Ajouter colonnes utiles
# ---------------------------------------------------------------------
df_clean['num_tokens'] = df_clean['transcript'].apply(lambda x: len(str(x).split()))
df_clean = df_clean[df_clean['num_tokens'] > 0].copy()

# Normalisation
df_clean['latency_per_token'] = df_clean['latency_sec'] / df_clean['num_tokens']
df_clean['memory_per_token'] = df_clean['memory_mb'] / df_clean['num_tokens']
df_clean['wer_per_token'] = df_clean['wer'] / df_clean['num_tokens']
df_clean['latency_per_sec'] = df_clean['latency_sec'] / df_clean['duration_sec']
df_clean['memory_per_sec'] = df_clean['memory_mb'] / df_clean['duration_sec']
df_clean['tokens_per_sec'] = df_clean['num_tokens'] / df_clean['duration_sec']

# Colonnes pour stats et boxplots
cols = ['latency_sec','memory_mb','wer','wer_token','levenshtein','levenshtein_pct',
        'accuracy','bleu3','meteor','chrf','rougeL','latency_per_sec','memory_per_sec',
        'tokens','tokens_per_sec','latency_per_token','memory_per_token','wer_per_token']

# ---------------------------------------------------------------------
# Statistiques descriptives par modèle
# ---------------------------------------------------------------------
stats = {}
for model_name, group_df in df_clean.groupby('model'):
    model_dir = os.path.join(REPORTING_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    agg_dict = {}
    for col in cols:
        if col in group_df:
            col_data = group_df[col].dropna()
            agg_dict[col] = {
                'count': col_data.count(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                '25%': col_data.quantile(0.25),
                '50%': col_data.median(),
                '75%': col_data.quantile(0.75),
                'max': col_data.max(),
                'skew': skew(col_data) if np.var(col_data) > 1e-12 else np.nan,
                'kurtosis': kurtosis(col_data) if np.var(col_data) > 1e-12 else np.nan
            }
    stats_df = pd.DataFrame(agg_dict).T
    stats_df.to_csv(os.path.join(model_dir, f"{model_name}_stats_medecin.csv"))
    stats[model_name] = stats_df

# ---------------------------------------------------------------------
# Boxplots par modèle avec moyenne en rouge
# ---------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_palette("pastel")

for col in cols:
    if col not in df_clean.columns:
        continue

    # Filtrer les lignes avec des valeurs non-NaN pour cette colonne
    df_plot = df_clean[['model', col]].dropna()
    if df_plot.empty:
        continue

    plt.figure(figsize=(12,6))
    ax = sns.boxplot(
        x='model',
        y=col,
        data=df_plot,
        showmeans=True,
        meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":8}
    )
    plt.title(f"{col} par modèle", fontsize=14)
    plt.ylabel(col, fontsize=12)
    plt.xlabel("Modèle", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Sauvegarde dans chaque sous-dossier modèle
    for model_name in df_plot['model'].unique():
        model_dir = os.path.join(REPORTING_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        plt.savefig(os.path.join(model_dir, f"boxplot_{col}_medecin.png"))
    plt.close()

print(f"Statistiques et boxplots sauvegardés dans {REPORTING_DIR}")
