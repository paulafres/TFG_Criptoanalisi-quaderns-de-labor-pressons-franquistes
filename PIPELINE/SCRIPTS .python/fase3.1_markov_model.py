"""
FASE 3.1: MODEL DE MARKOV BIDIRECCIONAL
========================================

Calcula probabilitats observacionals pures sobre el corpus de tokens
expandits, sense cap suposició semàntica:

  - P(t)                        : probabilitat marginal global
  - P(B | A)                    : probabilitat de successor
  - P(A | B)                    : probabilitat de predecessor (NOU)
  - P(t | inici_volta)          : probabilitat de ser primer token  (NOU)
  - P(t | final_volta)          : probabilitat de ser últim token   (NOU)
  - H(B | A)                    : entropia condicional de successors
  - H(A | B)                    : entropia condicional de predecessors

Entrada:  res_fase2/instruction_dataset.json
Sortida:  res_fase3.1/markov_model.json
"""

import os
import json
import math
from collections import Counter, defaultdict

from config import RES_FASE_3_1, ensure_dir, F2_INSTRUCTION_DATASET, F3_1_MARKOV_MODEL

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

ensure_dir(RES_FASE_3_1)

# ─────────────────────────────────────────────
# UTILITATS
# ─────────────────────────────────────────────

def entropia_shannon(distribucio):
    """H = -Σ p·log2(p) sobre una distribució (dict token→prob normalitzada)."""
    h = 0.0
    for p in distribucio.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


# ─────────────────────────────────────────────
# LECTURA
# ─────────────────────────────────────────────

with open(F2_INSTRUCTION_DATASET, "r", encoding="utf-8") as f:
    instruction_dataset = json.load(f)

# ─────────────────────────────────────────────
# RECOMPTES OBSERVACIONALS
# ─────────────────────────────────────────────

freq_global       = Counter()              # P(t)
freq_successors   = defaultdict(Counter)   # comptes[A][B] = #(A→B)
freq_predecessors = defaultdict(Counter)   # comptes[B][A] = #(A→B)
freq_inici        = Counter()              # #(t == primer token de la volta)
freq_final        = Counter()              # #(t == últim  token de la volta)

num_voltes = 0

for inst in instruction_dataset:
    toks = inst.get("tokens_expandits", [])
    if not toks:
        continue

    num_voltes += 1

    # marginals globals
    for tok in toks:
        freq_global[tok] += 1

    # posicions inici/final
    freq_inici[toks[0]]  += 1
    freq_final[toks[-1]] += 1

    # bigrames bidireccionals
    for i in range(len(toks) - 1):
        a, b = toks[i], toks[i + 1]
        freq_successors[a][b]   += 1
        freq_predecessors[b][a] += 1

# ─────────────────────────────────────────────
# CONSTRUCCIÓ DEL MODEL
# ─────────────────────────────────────────────

total_tokens = sum(freq_global.values())
markov_model = {}

for tok, count in freq_global.most_common():

    # P(t)
    p_global = count / total_tokens

    # P(t | inici_volta)  i  P(t | final_volta)
    p_inici = freq_inici[tok]  / num_voltes if num_voltes else 0.0
    p_final = freq_final[tok]  / num_voltes if num_voltes else 0.0

    # P(B | A=tok)
    succs_raw  = freq_successors.get(tok, {})
    total_succ = sum(succs_raw.values())
    succs_prob = (
        {b: c / total_succ for b, c in succs_raw.items()}
        if total_succ > 0 else {}
    )

    # P(A | B=tok)
    preds_raw  = freq_predecessors.get(tok, {})
    total_pred = sum(preds_raw.values())
    preds_prob = (
        {a: c / total_pred for a, c in preds_raw.items()}
        if total_pred > 0 else {}
    )

    # entropies condicionals
    h_succ = entropia_shannon(succs_prob)
    h_pred = entropia_shannon(preds_prob)

    markov_model[tok] = {
        "freq_absoluta": count,
        "P_global":      round(p_global, 6),
        "n_inici_volta": int(freq_inici[tok]),
        "n_final_volta": int(freq_final[tok]),
        "P_inici_volta": round(p_inici,  6),
        "P_final_volta": round(p_final,  6),
        "num_successors_distints":   len(succs_prob),
        "num_predecessors_distints": len(preds_prob),
        "H_successors":   round(h_succ, 4),
        "H_predecessors": round(h_pred, 4),
        # counts bruts (reutilitzats per Fase 3.1b, 3.2, ...)
        "counts_successors":   dict(
            sorted(succs_raw.items(), key=lambda x: -x[1])
        ),
        "counts_predecessors": dict(
            sorted(preds_raw.items(), key=lambda x: -x[1])
        ),
        # probabilitats normalitzades (reutilitzades per fases gramaticals)
        "transicions_successors":   {
            b: round(p, 6) for b, p in
            sorted(succs_prob.items(), key=lambda x: -x[1])
        },
        "transicions_predecessors": {
            a: round(p, 6) for a, p in
            sorted(preds_prob.items(), key=lambda x: -x[1])
        },
    }

# ─────────────────────────────────────────────
# METADADES GLOBALS DEL MODEL
# ─────────────────────────────────────────────

metadades = {
    "num_voltes":          num_voltes,
    "num_tokens_corpus":   total_tokens,
    "num_tokens_unics":    len(freq_global),
    "num_bigrames_unics":  sum(len(v) for v in freq_successors.values()),
    "num_bigrames_totals": sum(sum(v.values()) for v in freq_successors.values()),
}

output = {
    "metadades":     metadades,
    "markov_model":  markov_model,
}

# ─────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────

out_path = F3_1_MARKOV_MODEL
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("FASE 3.1 COMPLETADA")
print(f"  Voltes analitzades : {num_voltes}")
print(f"  Tokens al corpus   : {total_tokens}")
print(f"  Tokens únics       : {len(freq_global)}")
print(f"  Bigrames únics     : {metadades['num_bigrames_unics']}")
print(f"  Output             : {out_path}")
