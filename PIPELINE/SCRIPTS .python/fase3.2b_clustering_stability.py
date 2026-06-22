"""
FASE 3.2b: ESTABILITAT DEL CLUSTERING (BOOTSTRAP ARI)
======================================================

Valida la robustesa de les classes distribucionals de la Fase 3.2
mitjançant bootstrap no-paramètric sobre les voltes:

  1. Es carrega l'etiquetatge "ground-truth" de fase3.2.
  2. Per cada iteració de bootstrap:
      · Es re-mostregen N voltes amb reemplaçament (N = nº voltes original).
      · Es reconstrueixen tots els recomptes (bigrames, posicions, perfil
        per teixit) i la matriu PPMI smoothed (via lib compartida).
      · Es re-cluster amb el mateix k òptim de l'original.
      · Es calcula l'Adjusted Rand Index (Hubert & Arabie 1985) entre
        les etiquetes bootstrap i les originals, restringit als tokens
        clusteritzats en ambdues solucions.
  3. S'agreguen els ARI i s'informa de mitjana, sd i IC 95%.

Aquest fitxer és additiu i no modifica fase3.2. Comparteix amb fase3.2 la
mateixa lògica de PPMI + features (mòdul `fase3_lib_distributional`).

Entrades:
  · res_fase2/instruction_dataset.json
  · res_fase3.1/markov_model.json
  · res_fase3.2/distributional_classes.json

Sortida:
  · res_fase3.2b/clustering_stability.json
"""

from __future__ import annotations
import json, os, random, sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

# Lib compartida amb Fase 3.2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fase3_lib_distributional as fdlib

from config import (
    BASE, ensure_dir, RES_FASE_3_2B, RANDOM_STATE,
    F2_INSTRUCTION_DATASET, F3_1_MARKOV_MODEL, F3_2_DISTRIB_CLASSES,
    F3_2B_CLUSTERING_STABILITY,
)

# ─── Config ─────────────────────────────────────────────
IN_DATASET = F2_INSTRUCTION_DATASET
IN_MARKOV  = F3_1_MARKOV_MODEL
IN_CLASSES = F3_2_DISTRIB_CLASSES
OUT_JSON   = F3_2B_CLUSTERING_STABILITY
ensure_dir(RES_FASE_3_2B)

N_BOOTSTRAP       = 200
HAPAX_THRESHOLD   = 1
PPMI_SMOOTH_ALPHA = 0.75

rng = random.Random(RANDOM_STATE)

# ─── Lectura ────────────────────────────────────────────
with open(IN_DATASET, "r", encoding="utf-8") as f:
    instructions = json.load(f)
with open(IN_MARKOV, "r", encoding="utf-8") as f:
    markov_data = json.load(f)
with open(IN_CLASSES, "r", encoding="utf-8") as f:
    classes_data = json.load(f)

vocab_full_ref = sorted(markov_data["markov_model"].keys())
V_full = len(vocab_full_ref)
tok_index_full = {t: i for i, t in enumerate(vocab_full_ref)}

# Etiquetes originals (només tokens clusteritzats, sense C_RARE)
classes_info = classes_data["classes"]
labels_orig: dict[str, str] = {}
for cname, cinfo in classes_info.items():
    if cname == "C_RARE":
        continue
    for tok in cinfo["membres"]:
        labels_orig[tok] = cname
best_k = sum(1 for c in classes_info if c != "C_RARE")

print(f"Original: V_full={V_full}, k={best_k}, tokens clusteritzats={len(labels_orig)}")

# ─── Funció: pipeline complet sobre una mostra de voltes ─
def cluster_from_sample(sample_insts: list, k: int, seed: int) -> dict[str, int]:
    """
    Construeix la representació PPMI + features i clusteritza.
    Retorna mapping token → cluster_id (int). Tokens hapax NO inclosos.
    """
    counts = fdlib.build_corpus_counts(sample_insts)
    if counts.n_voltes == 0:
        return {}

    vocab_iter = sorted(t for t, c in counts.freq_token.items() if c > HAPAX_THRESHOLD)
    if len(vocab_iter) < k + 1:
        return {}

    marg = fdlib.compute_ppmi_marginals(
        counts.bigram_counts, tok_index_full, alpha=PPMI_SMOOTH_ALPHA
    )
    if marg.total_bigrams == 0:
        return {}

    # Estadístiques posicionals i d'entropia recalculades sobre la mostra
    n_v = counts.n_voltes
    p_inici = {t: counts.n_inici[t] / n_v for t in vocab_iter}
    p_final = {t: counts.n_final[t] / n_v for t in vocab_iter}
    h_succ  = {t: fdlib.entropia_shannon_counts(counts.bigram_counts.get(t, {}))
               for t in vocab_iter}
    # H(predecessors) requereix agregar les columnes del token
    preds_counts = {t: {} for t in vocab_iter}
    for a, d in counts.bigram_counts.items():
        for b, c in d.items():
            if b in preds_counts:
                preds_counts[b][a] = preds_counts[b].get(a, 0) + c
    h_pred = {t: fdlib.entropia_shannon_counts(preds_counts[t]) for t in vocab_iter}

    X = fdlib.build_feature_matrix(
        vocab_iter, vocab_full_ref, tok_index_full,
        counts, marg,
        p_inici=p_inici, p_final=p_final,
        h_succ=h_succ,   h_pred=h_pred,
    )

    X_norm = normalize(X, norm="l2", axis=1)
    n_comp = min(10, len(vocab_iter) - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    X_pca = pca.fit_transform(X_norm)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    lab = km.fit_predict(X_pca)
    return {vocab_iter[i]: int(lab[i]) for i in range(len(vocab_iter))}

# ─── Bucle bootstrap ────────────────────────────────────
N_voltes = len(instructions)
ari_values = []
n_intersect_values = []

for b in range(N_BOOTSTRAP):
    sample = [instructions[rng.randrange(N_voltes)] for _ in range(N_voltes)]
    labels_b = cluster_from_sample(sample, best_k, seed=RANDOM_STATE + b)
    if not labels_b:
        continue
    # Intersecció de tokens
    common = sorted(set(labels_orig.keys()) & set(labels_b.keys()))
    if len(common) < best_k + 1:
        continue
    y_true = [labels_orig[t] for t in common]
    y_pred = [labels_b[t]    for t in common]
    ari = adjusted_rand_score(y_true, y_pred)
    ari_values.append(ari)
    n_intersect_values.append(len(common))
    if (b + 1) % 25 == 0:
        print(f"  bootstrap {b+1}/{N_BOOTSTRAP}: ARI mitjà fins ara = "
              f"{np.mean(ari_values):.3f} (n_iter_vàlid={len(ari_values)})")

# ─── Estadístics ────────────────────────────────────────
ari_arr = np.array(ari_values)
mean_ari = float(ari_arr.mean())
std_ari  = float(ari_arr.std(ddof=1))
ci_lo    = float(np.percentile(ari_arr, 2.5))
ci_hi    = float(np.percentile(ari_arr, 97.5))

# ─── Sortida ────────────────────────────────────────────
out_doc = {
    "parametres": {
        "N_BOOTSTRAP":       N_BOOTSTRAP,
        "HAPAX_THRESHOLD":   HAPAX_THRESHOLD,
        "PPMI_SMOOTH_ALPHA": PPMI_SMOOTH_ALPHA,
        "RANDOM_STATE":      RANDOM_STATE,
        "k_fixat":           best_k,
        "metric":            "Adjusted Rand Index (Hubert & Arabie 1985)",
        "resampling":        "voltes amb reemplaçament",
    },
    "resultats": {
        "n_iteracions_valides": len(ari_values),
        "ARI_mitja":            round(mean_ari, 4),
        "ARI_sd":               round(std_ari,  4),
        "ARI_IC95":             [round(ci_lo, 4), round(ci_hi, 4)],
        "n_tokens_intersect_mitja": round(float(np.mean(n_intersect_values)), 2),
    },
    "ari_distribucio": [round(x, 4) for x in ari_values],
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_doc, f, indent=2, ensure_ascii=False)

print("\nFASE 3.2b COMPLETADA")
print(f"  Iteracions vàlides : {len(ari_values)}/{N_BOOTSTRAP}")
print(f"  ARI mitjà          : {mean_ari:.4f} ± {std_ari:.4f}")
print(f"  IC 95%             : [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Tokens intersect mitj. : {np.mean(n_intersect_values):.1f}")
print(f"  Output             : {OUT_JSON.relative_to(BASE.parent)}")
