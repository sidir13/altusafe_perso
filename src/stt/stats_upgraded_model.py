import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import numpy as np
from src.common.config import REPORTING_DIR, RESULTS_DIR

# ---------------------------------------------------------------------
# Lecture du CSV
# ---------------------------------------------------------------------
BENCHMARK_CSV = os.path.join(RESULTS_DIR, "stt_benchmark_medecin.csv")
df = pd.read_csv(BENCHMARK_CSV)

# ---------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------
df['memory_mb'] = df['memory_mb'].apply(lambda x: max(0, x))

num_cols = ['latency_sec', 'memory_mb', 'wer', 'accuracy']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=num_cols).copy()

# Créer dossier reporting
os.makedirs(REPORTING_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Statistiques descriptives globales
# ---------------------------------------------------------------------
agg_dict = {}
for col in num_cols:
    col_data = df_clean[col].dropna()
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
stats_csv_path = os.path.join(REPORTING_DIR, "stats_medecin.csv")
stats_df.to_csv(stats_csv_path)
print(f"Statistiques globales sauvegardées dans : {stats_csv_path}")

# ---------------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_palette("pastel")

for col in num_cols:
    df_plot = df_clean[[col]].dropna()
    if df_plot.empty:
        continue

    plt.figure(figsize=(8,5))
    ax = sns.boxplot(
        y=col,
        data=df_plot,
        showmeans=True,
        meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"red","markersize":8}
    )
    plt.title(f"{col} global", fontsize=14)
    plt.ylabel(col, fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.savefig(os.path.join(REPORTING_DIR, f"boxplot_{col}_medecin_upgraded.png"))
    plt.close()

print(f"Boxplots sauvegardés dans : {REPORTING_DIR}")
