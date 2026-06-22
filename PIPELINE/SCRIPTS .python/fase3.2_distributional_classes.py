"""
FASE 3.2: CLASSES DISTRIBUCIONALS EMERGENTS
============================================

Agrupa els tokens en classes funcionals **sense cap suposició semàntica**,
basant-se únicament en el seu context observat (Harris 1954, distributional
semantics).

Per cada token construeix un vector de característiques purament
observacionals:
  - successors / predecessors  → pesos PPMI (Levy & Goldberg 2014),
                                  no probabilitats brutes, per evitar que
                                  els tokens molt freqüents (p. ex. `l`)
                                  contaminin la similitud
  - posicional (P_inici, P_final, H_succ, H_pred, mean_pos, std_pos)
  - estructural (es_macro, profunditat, longitud)
  - perfil per teixit (distribució relativa entre teixits)

Política d'hapax: els tokens amb `freq_absoluta <= HAPAX_THRESHOLD`
no entren al clustering (context insuficient → degraden la silhouette).
S'assignen automàticament a la classe `C_RARE` i es documenten al JSON
de sortida.

Nota: PPMI ja penalitza per ell mateix les transicions amb lift<1 (les
posa a zero), per tant **no apliquem cap filtre extra** basat en
significança FDR. El test de la Fase 3.1b s'aprofita aigües avall a la
Fase 5 (anomalies estructurals).

Reducció amb PCA, agrupament amb K-Means escollint k òptim per Silhouette.

Entrada:
  - res_fase3.1/markov_model.json
  - res_fase2/instruction_dataset.json

Sortida:
  - res_fase3.2/distributional_classes.json
"""

import os
import json
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Lib compartida amb Fase 3.2b
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fase3_lib_distributional as fdlib

from config import (
    RES_FASE_3_2, ensure_dir, RANDOM_STATE,
    F2_INSTRUCTION_DATASET, F3_1_MARKOV_MODEL, F3_2_DISTRIB_CLASSES,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

K_MIN, K_MAX       = 2, 18    # rang per cercar k òptim
HAPAX_THRESHOLD    = 1        # freq <= aquest llindar → classe RARE
PPMI_SMOOTH_ALPHA  = 0.75     # context distribution smoothing (Levy & Goldberg 2014)

ensure_dir(RES_FASE_3_2)

# ─────────────────────────────────────────────
# CARREGA
# ─────────────────────────────────────────────

with open(F3_1_MARKOV_MODEL, "r", encoding="utf-8") as f:
    markov_data = json.load(f)
markov = markov_data["markov_model"]

with open(F2_INSTRUCTION_DATASET, "r", encoding="utf-8") as f:
    instructions = json.load(f)

vocab_full = sorted(markov.keys())
V_full = len(vocab_full)
tok_index_full = {t: i for i, t in enumerate(vocab_full)}

# ─────────────────────────────────────────────
# RECOMPTES
# ─────────────────────────────────────────────

counts = fdlib.build_corpus_counts(instructions)

T = len(counts.teixits_list)

# ─────────────────────────────────────────────
# SEPARACIÓ HAPAX vs CLUSTERABLES
# ─────────────────────────────────────────────

freq_token = {t: markov[t]["freq_absoluta"] for t in vocab_full}
vocab_rare  = [t for t in vocab_full if freq_token[t] <= HAPAX_THRESHOLD]
vocab       = [t for t in vocab_full if freq_token[t] >  HAPAX_THRESHOLD]
V = len(vocab)
tok_index = {t: i for i, t in enumerate(vocab)}

# ─────────────────────────────────────────────
# MATRIU PPMI + FEATURES (via lib compartida)
# ─────────────────────────────────────────────

marg = fdlib.compute_ppmi_marginals(
    counts.bigram_counts, tok_index_full, alpha=PPMI_SMOOTH_ALPHA
)

p_inici = {t: markov[t]["P_inici_volta"]  for t in vocab_full}
p_final = {t: markov[t]["P_final_volta"]  for t in vocab_full}
h_succ  = {t: markov[t]["H_successors"]   for t in vocab_full}
h_pred  = {t: markov[t]["H_predecessors"] for t in vocab_full}

X = fdlib.build_feature_matrix(
    vocab, vocab_full, tok_index_full,
    counts, marg,
    p_inici=p_inici, p_final=p_final,
    h_succ=h_succ,   h_pred=h_pred,
)

# Normalització L2 per fila
X_norm = normalize(X, norm="l2", axis=1)

# ─────────────────────────────────────────────
# REDUCCIÓ DE DIMENSIONALITAT (PCA)
# ─────────────────────────────────────────────

n_components = min(10, V - 1)
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_norm)
X_2d  = X_pca[:, :2]

# ─────────────────────────────────────────────
# CLUSTERING — KMeans amb k òptim per Silhouette
# ─────────────────────────────────────────────

resultats_k = []
for k in range(K_MIN, min(K_MAX, V - 1) + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels_k = km.fit_predict(X_pca)
    if len(set(labels_k)) < 2:
        continue
    score = silhouette_score(X_pca, labels_k)
    resultats_k.append((k, score, labels_k, km))

best_k, best_score, best_labels, best_km = max(resultats_k, key=lambda x: x[1])

# ─────────────────────────────────────────────
# OUTLIERS intra-clúster (z > 2)
# ─────────────────────────────────────────────

centroides = best_km.cluster_centers_
dist_centroide = np.linalg.norm(X_pca - centroides[best_labels], axis=1)

outlier_score = np.zeros(V)
es_outlier = np.zeros(V, dtype=bool)
for c in range(best_k):
    mask = best_labels == c
    d = dist_centroide[mask]
    mu, sd = d.mean(), d.std()
    if sd > 0:
        z = (d - mu) / sd
        outlier_score[mask] = z
        es_outlier[mask] = z > 2.0

# ─────────────────────────────────────────────
# CARACTERITZACIÓ DE CADA CLASSE
# ─────────────────────────────────────────────

classes_info = {}
for c in range(best_k):
    membres = [vocab[i] for i in range(V) if best_labels[i] == c]
    idx = [i for i in range(V) if best_labels[i] == c]

    mean_pos_vals = [X[i, 2 * V_full + 4] for i in idx]
    h_succ_vals   = [X[i, 2 * V_full + 2] for i in idx]
    p_inici_vals  = [X[i, 2 * V_full + 0] for i in idx]
    p_final_vals  = [X[i, 2 * V_full + 1] for i in idx]
    frac_macro    = float(np.mean([X[i, 2 * V_full + 6] for i in idx]))

    classes_info[f"C{c}"] = {
        "num_tokens": len(membres),
        "membres": membres,
        "centroide_2d": [round(float(centroides[c, 0]), 4),
                         round(float(centroides[c, 1]), 4)],
        "mean_pos_norm_mitja":  round(float(np.mean(mean_pos_vals)), 4),
        "H_successors_mitja":   round(float(np.mean(h_succ_vals)), 4),
        "P_inici_volta_mitja":  round(float(np.mean(p_inici_vals)), 4),
        "P_final_volta_mitja":  round(float(np.mean(p_final_vals)), 4),
        "frac_macros":          round(frac_macro, 4),
    }

# Classe sintètica per a tokens hapax (no s'han clusteritzat)
if vocab_rare:
    classes_info["C_RARE"] = {
        "num_tokens": len(vocab_rare),
        "membres": vocab_rare,
        "centroide_2d": [None, None],
        "mean_pos_norm_mitja": None,
        "H_successors_mitja": None,
        "P_inici_volta_mitja": None,
        "P_final_volta_mitja": None,
        "frac_macros": None,
        "nota": (
            f"Tokens amb freq_absoluta <= {HAPAX_THRESHOLD}. "
            "Context insuficient per clustering distribucional fiable. "
            "Agrupats en una sola classe d'equivalència 'context-pobre'."
        ),
    }

# ─────────────────────────────────────────────
# SORTIDA: tokens (clusteritzats + RARE) i classes
# ─────────────────────────────────────────────

tokens_out = {}
for i, tok in enumerate(vocab):
    tokens_out[tok] = {
        "classe":         f"C{int(best_labels[i])}",
        "outlier_score":  round(float(outlier_score[i]), 4),
        "es_outlier":     bool(es_outlier[i]),
        "vector_2d":      [round(float(X_2d[i, 0]), 4),
                           round(float(X_2d[i, 1]), 4)],
        "freq_absoluta":  markov[tok]["freq_absoluta"],
    }
for tok in vocab_rare:
    tokens_out[tok] = {
        "classe":        "C_RARE",
        "outlier_score": None,
        "es_outlier":    False,
        "vector_2d":     [None, None],
        "freq_absoluta": markov[tok]["freq_absoluta"],
    }

output = {
    "parametres": {
        "K_MIN": K_MIN,
        "K_MAX": K_MAX,
        "RANDOM_STATE": RANDOM_STATE,
        "HAPAX_THRESHOLD": HAPAX_THRESHOLD,
        "PPMI_SMOOTH_ALPHA": PPMI_SMOOTH_ALPHA,
        "n_init_kmeans": 10,
        "outlier_z_threshold": 2.0,
    },
    "metadades": {
        "num_tokens_totals":     V_full,
        "num_tokens_clustered":  V,
        "num_tokens_rare":       len(vocab_rare),
        "num_teixits":           T,
        "dim_vector_total":      int(X.shape[1]),
        "n_components_pca":      n_components,
        "variancia_pca":         [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "k_provats":             [k for k, _, _, _ in resultats_k],
        "silhouette_per_k":      {int(k): round(float(s), 4) for k, s, _, _ in resultats_k},
        "k_optim":               int(best_k),
        "silhouette_optim":      round(float(best_score), 4),
    },
    "tokens":  tokens_out,
    "classes": classes_info,
}

out_json = F3_2_DISTRIB_CLASSES
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────
# REPORT CONSOLA
# ─────────────────────────────────────────────

print("FASE 3.2 COMPLETADA")
print(f"  Tokens totals        : {V_full}")
print(f"  Tokens clusteritzats : {V}  (freq > {HAPAX_THRESHOLD})")
print(f"  Tokens RARE          : {len(vocab_rare)}")
print(f"  Dimensions vector    : {X.shape[1]}  (PPMI smoothed)")
print(f"  Variància PCA (top5) : {[round(float(v), 3) for v in pca.explained_variance_ratio_[:5]]}")
print(f"  k òptim (silhouette) : {best_k}  (score={best_score:.4f})")
print(f"  Outliers (z>2)       : {int(es_outlier.sum())}")
print()
print("Resum per classe:")
for cname, cinfo in classes_info.items():
    if cname == "C_RARE":
        print(f"  {cname} ({cinfo['num_tokens']:>2} toks)  [hapax, no clusteritzats]  "
              f"membres={cinfo['membres'][:8]}"
              + (" ..." if cinfo['num_tokens'] > 8 else ""))
    else:
        print(f"  {cname}  ({cinfo['num_tokens']:>2} toks)  pos_mitja={cinfo['mean_pos_norm_mitja']:.2f}  "
              f"H_succ={cinfo['H_successors_mitja']:.2f}  membres={cinfo['membres'][:8]}"
              + (" ..." if cinfo['num_tokens'] > 8 else ""))
print()
print(f"  JSON: {out_json}")
