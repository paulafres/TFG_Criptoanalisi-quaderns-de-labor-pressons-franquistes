"""
FASE 3.1b: SIGNIFICANÇA ESTADÍSTICA DE TRANSICIONS
====================================================

Per a cada bigrama observat (A→B) amb una freqüència mínima, contrasta
si la transició és més probable del que s'esperaria sota independència
(H0: P(B|A) = P(B)).

Mètode:
  · Taula de contingència 2×2 per cada bigrama
        ┌──────────┬──────────┐
        │ succ = B │ succ ≠ B │
   ┌────┼──────────┼──────────┤
   │  A │   n_AB   │ n_A_notB │
   │ ¬A │  n_notA_B│n_notA_notB│
   └────┴──────────┴──────────┘
  · Test exacte de Fisher (one-sided, alternative='greater').
  · Correcció Benjamini-Hochberg (FDR) sobre tots els p-valors.
  · Odds ratio i lift = P(B|A) / P(B) com a magnitud d'efecte.

Aquest fitxer és additiu i no modifica fase3.1.

Reutilitza els recomptes de bigrames ja calculats a Fase 3.1
(camps `counts_successors` del markov_model), evitant re-traversar
el corpus i garantint coherència numèrica.

Entrada:  res_fase3.1/markov_model.json
Sortida:  res_fase3.1b/transition_significance.json
           res_fase3.1b/transition_significance.csv
"""

from __future__ import annotations
import os, json, csv
from collections import Counter
from pathlib import Path
from scipy.stats import fisher_exact

from config import (
    BASE, RES_FASE_3_1B, ensure_dir, RANDOM_STATE,
    F3_1_MARKOV_MODEL,
)

# ─── Config ───────────────────────────────────
IN_MARKOV  = F3_1_MARKOV_MODEL
OUT_DIR    = RES_FASE_3_1B
OUT_JSON   = OUT_DIR / "transition_significance.json"
OUT_CSV    = OUT_DIR / "transition_significance.csv"

MIN_COUNT_BIGRAM = 3        # bigrames amb menys ocurrències s'ignoren (soroll)
ALPHA_FDR        = 0.05     # llindar FDR

ensure_dir(OUT_DIR)

# ─── Lectura del model de Markov ───────────────────────
with open(IN_MARKOV, "r", encoding="utf-8") as f:
    markov_doc = json.load(f)

markov_model    = markov_doc["markov_model"]
N               = markov_doc["metadades"]["num_bigrames_totals"]

# Recompostes a partir dels counts ja persistits a Fase 3.1:
#   freq_successors[A][B] = counts_successors del token A
freq_successors = {
    a: dict(node.get("counts_successors", {}))
    for a, node in markov_model.items()
    if node.get("counts_successors")
}

# nB = #(transicions on B apareix com a successor), sumant per columnes
n_succ_B = Counter()
for a, bs in freq_successors.items():
    for b, c in bs.items():
        n_succ_B[b] += c

# ─── Fisher exact per bigrama ──────────────────────────
# Sigui:
#   N   = total de bigrames del corpus
#   nA  = #(transicions amb successor partint de A) = Σ_b n[A][b]
#   nB  = #(transicions on apareix B com a successor)
results = []

for a, bs in freq_successors.items():
    nA = sum(bs.values())                # files = A
    for b, n_AB in bs.items():
        if n_AB < MIN_COUNT_BIGRAM:
            continue
        nB        = n_succ_B[b]          # columnes = B
        n_A_notB  = nA - n_AB
        n_notA_B  = nB - n_AB
        n_notA_notB = N - nA - n_notA_B
        # Salvaguarda numèrica
        if min(n_AB, n_A_notB, n_notA_B, n_notA_notB) < 0:
            continue
        table = [[n_AB, n_A_notB], [n_notA_B, n_notA_notB]]
        try:
            odds, p_one_sided = fisher_exact(table, alternative="greater")
        except ValueError:
            continue
        # Lift = P(B|A) / P(B)
        p_B_given_A = n_AB / nA if nA else 0.0
        p_B         = nB  / N  if N  else 0.0
        lift        = (p_B_given_A / p_B) if p_B > 0 else float("inf")
        results.append({
            "A": a,
            "B": b,
            "n_AB": n_AB,
            "P(B|A)": round(p_B_given_A, 6),
            "P(B)":   round(p_B, 6),
            "lift":   round(lift, 4) if lift != float("inf") else None,
            "odds_ratio":  round(float(odds), 4) if odds != float("inf") else None,
            "p_value":     float(p_one_sided),
        })

# ─── Correcció FDR (Benjamini-Hochberg) ────────────────
m = len(results)
results_sorted = sorted(results, key=lambda r: r["p_value"])
for rank, r in enumerate(results_sorted, start=1):
    r["q_value_BH"] = min(1.0, r["p_value"] * m / rank)
# Monotonia ascendent (forçar q no decreixent)
min_q = 1.0
for r in reversed(results_sorted):
    min_q = min(min_q, r["q_value_BH"])
    r["q_value_BH"] = round(min_q, 8)
    r["significatiu_FDR"] = r["q_value_BH"] < ALPHA_FDR
    r["p_value"] = round(r["p_value"], 8)

# Re-ordenar finalment per lift descendent dins els significatius
results_sorted.sort(key=lambda r: (-1 if r["significatiu_FDR"] else 0,
                                   -(r["lift"] or 0),
                                   r["p_value"]))

# ─── Sortides ──────────────────────────────────────────
n_signif = sum(1 for r in results_sorted if r["significatiu_FDR"])
out_doc = {
    "parametres": {
        "MIN_COUNT_BIGRAM": MIN_COUNT_BIGRAM,
        "ALPHA_FDR":        ALPHA_FDR,
        "test":             "fisher_exact (one-sided, alternative='greater')",
        "correccio":        "Benjamini-Hochberg (FDR)",
        "RANDOM_STATE":     RANDOM_STATE,
    },
    "metadades": {
        "num_bigrames_testats":      m,
        "num_significatius_FDR":     n_signif,
        "total_bigrames_corpus":     N,
        "min_count_filtrat":         MIN_COUNT_BIGRAM,
    },
    "transicions": results_sorted,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_doc, f, indent=2, ensure_ascii=False)

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["A", "B", "n_AB", "P(B|A)", "P(B)", "lift",
                "odds_ratio", "p_value", "q_value_BH", "significatiu_FDR"])
    for r in results_sorted:
        w.writerow([r["A"], r["B"], r["n_AB"], r["P(B|A)"], r["P(B)"],
                    r["lift"], r["odds_ratio"], r["p_value"],
                    r["q_value_BH"], r["significatiu_FDR"]])

print("FASE 3.1b COMPLETADA")
print(f"  Bigrames testats        : {m}")
print(f"  Significatius (q<{ALPHA_FDR}) : {n_signif}")
print(f"  Top-10 per lift entre significatius:")
top = [r for r in results_sorted if r["significatiu_FDR"]][:10]
for r in top:
    print(f"    {r['A']:>12} → {r['B']:<12}  n={r['n_AB']:>4}  "
          f"lift={r['lift'] or 0:>7.2f}  q={r['q_value_BH']:.2e}")
print(f"  JSON : {OUT_JSON.relative_to(BASE.parent)}")
print(f"  CSV  : {OUT_CSV.relative_to(BASE.parent)}")
