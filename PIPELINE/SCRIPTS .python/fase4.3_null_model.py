"""
FASE 4.3: MODEL NUL PER PERMUTACIÓ
====================================

Contrasta si l'estructura seqüencial observada al corpus és significativa
respecte d'un model nul que destrueix l'ordre però manté:
  · les freqüències marginals de cada token,
  · la longitud de cada volta,
  · el repartiment per teixit.

Hipòtesi nul·la H0: l'ordre dels tokens dins cada volta és intercanviable.

Estadístic: entropia condicional global del Markov de successors,
            H_global = Σ_a P(a) · H(B | A=a)
Si l'observat << null  ⇒ l'ordre conté informació predictiva
(rebutgem H0 amb un test unilateral inferior).

Mètode:
  1. H_obs es calcula sobre l'ordre original.
  2. Per N_PERM permutacions, es barreja l'ordre dels tokens DINS de
     cada volta i es recalcula H_perm.
  3. p_emp = (#{H_perm ≤ H_obs} + 1) / (N_PERM + 1)
  4. z = (H_obs - mean_perm) / sd_perm

Entrada:  res_fase2/instruction_dataset.json
Sortida:  res_fase4.3/null_model_permutation.json
"""

from __future__ import annotations
import json, math, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from config import (
    BASE, RES_FASE_4_3, ensure_dir, RANDOM_STATE,
    F2_INSTRUCTION_DATASET,
)

IN_DATASET = F2_INSTRUCTION_DATASET
OUT_JSON   = RES_FASE_4_3 / "null_model_permutation.json"
ensure_dir(RES_FASE_4_3)

N_PERM        = 1000

rng = random.Random(RANDOM_STATE)

with open(IN_DATASET, "r", encoding="utf-8") as f:
    instructions = json.load(f)

voltes = [inst.get("tokens_expandits", []) for inst in instructions
          if inst.get("tokens_expandits")]

def h_global_from_voltes(voltes_list: list[list[str]]) -> float:
    """Calcula H_global = Σ_a P(a) · H(B|A=a) a partir d'una llista de voltes."""
    freq_a = Counter()
    succ = defaultdict(Counter)
    for v in voltes_list:
        for i, tok in enumerate(v):
            if i + 1 < len(v):
                freq_a[tok] += 1
                succ[tok][v[i + 1]] += 1
    total = sum(freq_a.values())
    if total == 0:
        return 0.0
    H = 0.0
    for a, c_a in freq_a.items():
        p_a = c_a / total
        tot = sum(succ[a].values())
        if tot == 0:
            continue
        h_a = 0.0
        for c in succ[a].values():
            p = c / tot
            if p > 0:
                h_a -= p * math.log2(p)
        H += p_a * h_a
    return H

H_obs = h_global_from_voltes(voltes)
print(f"H_obs (corpus original) = {H_obs:.4f} bits")

H_perm_values = np.empty(N_PERM, dtype=np.float64)
for p in range(N_PERM):
    permuted = []
    for v in voltes:
        v_copy = v[:]
        rng.shuffle(v_copy)
        permuted.append(v_copy)
    H_perm_values[p] = h_global_from_voltes(permuted)
    if (p + 1) % 100 == 0:
        print(f"  perm {p+1}/{N_PERM}: H_perm mitjà = {H_perm_values[:p+1].mean():.4f}")

mean_p = float(H_perm_values.mean())
sd_p   = float(H_perm_values.std(ddof=1))
z_score = (H_obs - mean_p) / sd_p if sd_p > 0 else 0.0
p_emp_inf = float((np.sum(H_perm_values <= H_obs) + 1) / (N_PERM + 1))
ic_lo = float(np.percentile(H_perm_values, 2.5))
ic_hi = float(np.percentile(H_perm_values, 97.5))
delta = H_obs - mean_p
reduction = delta / mean_p if mean_p > 0 else 0.0

out_doc = {
    "parametres": {
        "N_PERM":       N_PERM,
        "RANDOM_STATE": RANDOM_STATE,
        "estadistic":   "H_global = Σ_a P(a) · H(B|A=a)  [Markov d'ordre 1]",
        "permutacio":   "shuffle de tokens dins de cada volta (manté freqs marginals + longituds)",
        "test":         "unilateral inferior (estructura predictiva ⇒ H_obs < H_null)",
    },
    "resultats": {
        "H_observat":              round(H_obs, 6),
        "H_null_mitja":            round(mean_p, 6),
        "H_null_sd":               round(sd_p, 6),
        "H_null_IC95":             [round(ic_lo, 6), round(ic_hi, 6)],
        "delta_obs_vs_null":       round(delta, 6),
        "reduccio_relativa":       round(reduction, 6),
        "z_score":                 round(z_score, 4),
        "p_value_empirica":        round(p_emp_inf, 6),
        "significatiu_α_0_05":     p_emp_inf < 0.05,
    },
    "H_perm_distribucio_resum": {
        "min":    round(float(H_perm_values.min()), 6),
        "max":    round(float(H_perm_values.max()), 6),
        "q05":    round(float(np.percentile(H_perm_values, 5)), 6),
        "q50":    round(float(np.percentile(H_perm_values, 50)), 6),
        "q95":    round(float(np.percentile(H_perm_values, 95)), 6),
    },
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out_doc, f, indent=2, ensure_ascii=False)

print("\nFASE 4.3 COMPLETADA")
print(f"  H_observat   : {H_obs:.4f} bits")
print(f"  H_null mitjà : {mean_p:.4f} ± {sd_p:.4f}")
print(f"  Reducció     : {100*reduction:.2f}% respecte al nul")
print(f"  z-score      : {z_score:.2f}")
print(f"  p-valor emp  : {p_emp_inf:.4f}")
print(f"  Output       : {OUT_JSON.relative_to(BASE.parent)}")
