"""
FASE 4.4: LEAVE-ONE-TEIXIT-OUT CROSS-VALIDATION
=================================================

Quantifica la generalització del model bigrama (Markov d'ordre 1) entre
teixits diferents. Per cada teixit T:

  1. S'entrena un model bigrama sobre TOTS els altres teixits.
  2. Es calcula la surprisal mitjana per token sobre les voltes de T.
  3. Es compara amb la surprisal in-sample (entrenant sobre tot el corpus).

Smoothing additiu (Laplace) amb constant α petita per evitar surprisal
infinita en bigrames no vistos al fold d'entrenament:

      P(b | a) = (n_ab + α) / (n_a + α · V_train)

On V_train és la mida del vocabulari del fold d'entrenament.

Aquest mètode produeix una mesura de quant un teixit "sorprèn" si
només has llegit els altres deu — útil per identificar teixits
atípics i defensar la transferibilitat del model.

Entrada:  res_fase2/instruction_dataset.json
Sortida:  res_fase4.4/loo_cv_teixits.json
           res_fase4.4/loo_cv_teixits.csv
"""

from __future__ import annotations
import json, math, csv
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from config import (
    BASE, RES_FASE_4_4, ensure_dir, RANDOM_STATE,
    F2_INSTRUCTION_DATASET, F4_4_LOO_CV_JSON, F4_4_LOO_CV_CSV,
)

IN_DATASET = F2_INSTRUCTION_DATASET
OUT_JSON   = F4_4_LOO_CV_JSON
OUT_CSV    = F4_4_LOO_CV_CSV
ensure_dir(RES_FASE_4_4)

ALPHA_SMOOTH = 0.01           # smoothing additiu (Laplace lleuger)

with open(IN_DATASET, "r", encoding="utf-8") as f:
    instructions = json.load(f)

# Agrupar per teixit
voltes_per_teixit: dict[str, list[list[str]]] = defaultdict(list)
for inst in instructions:
    toks = inst.get("tokens_expandits", [])
    if toks:
        voltes_per_teixit[inst.get("teixit", "?")].append(toks)

teixits = sorted(voltes_per_teixit.keys())
N_teixits = len(teixits)
print(f"Teixits: {N_teixits} · voltes totals = {sum(len(v) for v in voltes_per_teixit.values())}")

# ─── Funció: entrenar bigrama + marginal ─────────────────
def train_bigram(corpus: list[list[str]]):
    freq_a    = Counter()
    freq_pair = defaultdict(Counter)
    freq_uni  = Counter()
    for v in corpus:
        for i, tok in enumerate(v):
            freq_uni[tok] += 1
            if i + 1 < len(v):
                freq_a[tok] += 1
                freq_pair[tok][v[i + 1]] += 1
    V = sorted(freq_uni.keys())
    return {
        "V": V,
        "freq_a": freq_a,
        "freq_pair": freq_pair,
        "freq_uni": freq_uni,
        "total_uni": sum(freq_uni.values()),
    }

def surprisal_volta(v: list[str], model: dict, alpha: float) -> tuple[float, int, int]:
    """Retorna (surprisal_total, n_tokens_avaluats, n_oov_bigrames)."""
    V_train = len(model["V"])
    s = 0.0
    n_eval = 0
    n_oov = 0
    # Token inicial: probabilitat unigrama (smoothing additiu sobre V_train)
    if v:
        c = model["freq_uni"].get(v[0], 0)
        p = (c + alpha) / (model["total_uni"] + alpha * V_train)
        s += -math.log2(p)
        n_eval += 1
    # Tokens següents: bigrama amb smoothing
    for i in range(1, len(v)):
        a = v[i - 1]
        b = v[i]
        n_a = model["freq_a"].get(a, 0)
        n_ab = model["freq_pair"].get(a, {}).get(b, 0)
        if n_ab == 0:
            n_oov += 1
        p = (n_ab + alpha) / (n_a + alpha * V_train)
        s += -math.log2(p)
        n_eval += 1
    return s, n_eval, n_oov

# ─── Surprisal in-sample (entrenat amb tot el corpus) ────
all_voltes = [v for tx in teixits for v in voltes_per_teixit[tx]]
model_full = train_bigram(all_voltes)
in_sample = {}
for tx in teixits:
    s_tot = n_tot = oov_tot = 0
    for v in voltes_per_teixit[tx]:
        s, n, o = surprisal_volta(v, model_full, ALPHA_SMOOTH)
        s_tot += s; n_tot += n; oov_tot += o
    in_sample[tx] = {
        "surprisal_mitjana": s_tot / n_tot if n_tot else 0.0,
        "n_tokens": n_tot,
        "n_oov_bigrames": oov_tot,
    }

# ─── LOO-CV ──────────────────────────────────────────────
results = []
for tx in teixits:
    train_corpus = [v for tx2 in teixits if tx2 != tx for v in voltes_per_teixit[tx2]]
    model_loo = train_bigram(train_corpus)
    s_tot = n_tot = oov_tot = 0
    for v in voltes_per_teixit[tx]:
        s, n, o = surprisal_volta(v, model_loo, ALPHA_SMOOTH)
        s_tot += s; n_tot += n; oov_tot += o
    surp_loo = s_tot / n_tot if n_tot else 0.0
    surp_in  = in_sample[tx]["surprisal_mitjana"]
    results.append({
        "teixit":               tx,
        "n_voltes":             len(voltes_per_teixit[tx]),
        "n_tokens_eval":        n_tot,
        "surprisal_loo":        round(surp_loo, 4),
        "surprisal_in_sample":  round(surp_in, 4),
        "delta_loo_vs_in":      round(surp_loo - surp_in, 4),
        "n_oov_bigrames_loo":   oov_tot,
        "frac_oov_loo":         round(oov_tot / n_tot, 4) if n_tot else 0.0,
    })

# Ordenar per surprisal LOO descendent (més atípics primer)
results.sort(key=lambda r: -r["surprisal_loo"])

surp_loo_vals = [r["surprisal_loo"] for r in results]
mean_loo = float(np.mean(surp_loo_vals))
std_loo  = float(np.std(surp_loo_vals, ddof=1))

out_doc = {
    "parametres": {
        "ALPHA_SMOOTH": ALPHA_SMOOTH,
        "RANDOM_STATE": RANDOM_STATE,
        "metode":       "Leave-one-teixit-out CV · bigrama amb smoothing additiu",
    },
    "metadades": {
        "n_teixits":               N_teixits,
        "surprisal_loo_mitjana":   round(mean_loo, 4),
        "surprisal_loo_sd":        round(std_loo, 4),
        "surprisal_in_sample_global": round(
            sum(r["surprisal_in_sample"] * r["n_tokens_eval"] for r in results) /
            sum(r["n_tokens_eval"] for r in results), 4),
    },
    "per_teixit": results,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_doc, f, indent=2, ensure_ascii=False)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["teixit", "n_voltes", "n_tokens_eval", "surprisal_loo",
                "surprisal_in_sample", "delta_loo_vs_in",
                "n_oov_bigrames_loo", "frac_oov_loo"])
    for r in results:
        w.writerow([r["teixit"], r["n_voltes"], r["n_tokens_eval"],
                    r["surprisal_loo"], r["surprisal_in_sample"],
                    r["delta_loo_vs_in"], r["n_oov_bigrames_loo"],
                    r["frac_oov_loo"]])

print("\nFASE 4.4 COMPLETADA")
print(f"  Surprisal LOO mitjana : {mean_loo:.4f} ± {std_loo:.4f} bits/token")
print(f"  Surprisal in-sample   : {out_doc['metadades']['surprisal_in_sample_global']:.4f}")
print(f"  Rànquing per atipicitat (LOO):")
print(f"    {'teixit':<22} {'n_v':>5} {'surp_LOO':>10} {'surp_in':>10} {'Δ':>8} {'%OOV':>8}")
for r in results:
    print(f"    {r['teixit']:<22} {r['n_voltes']:>5} "
          f"{r['surprisal_loo']:>10.4f} {r['surprisal_in_sample']:>10.4f} "
          f"{r['delta_loo_vs_in']:>+8.4f} {100*r['frac_oov_loo']:>7.2f}%")
print(f"  JSON : {OUT_JSON.relative_to(BASE.parent)}")
print(f"  CSV  : {OUT_CSV.relative_to(BASE.parent)}")
