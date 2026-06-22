"""
LLIBRERIA COMPARTIDA — REPRESENTACIÓ DISTRIBUCIONAL DE TOKENS
==============================================================

Funcions reutilitzades per Fase 3.2 (classes distribucionals) i Fase 3.2b
(estabilitat per bootstrap). Concentra la construcció de:

  · recomptes sobre una col·lecció d'instruccions (bigrames, posicions,
    perfil per teixit, freqüències marginals i de vora de volta);
  · marginals PPMI smoothed (Levy & Goldberg 2014);
  · vectors PPMI de successors / predecessors;
  · matriu de característiques unificada (PPMI + posicional + estructural +
    perfil teixit) amb el mateix layout que abans.

L'objectiu és garantir que ambdues fases comparteixin exactament la mateixa
geometria, eliminant duplicació i risc de divergència numèrica.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ─────────────────────────────────────────────
# RECOMPTES SOBRE INSTRUCCIONS
# ─────────────────────────────────────────────

@dataclass
class CorpusCounts:
    bigram_counts: dict       # dict[a][b] -> int
    positions_norm: dict      # dict[token] -> list[float]
    freq_per_teixit: dict     # dict[token] -> Counter[teixit]
    freq_token: Counter       # Counter[token]
    n_inici: Counter          # Counter[token]  (#vegades que és primer de volta)
    n_final: Counter          # Counter[token]  (#vegades que és últim de volta)
    n_voltes: int             # nombre de voltes amb contingut
    teixits_list: list        # llista ordenada de teixits observats


def build_corpus_counts(instructions: Iterable[dict]) -> CorpusCounts:
    """
    Construeix tots els recomptes que necessita el pipeline distribucional
    a partir d'una llista d'instruccions ({tokens_expandits, teixit, ...}).

    El bigram_counts retornat és un `defaultdict` per facilitar filtrats
    posteriors (p.ex. eliminar transicions no significatives).
    """
    bigram_counts   = defaultdict(lambda: defaultdict(int))
    positions_norm  = defaultdict(list)
    freq_per_teixit = defaultdict(Counter)
    freq_token      = Counter()
    n_inici         = Counter()
    n_final         = Counter()
    n_voltes        = 0
    teixits_set     = set()

    for inst in instructions:
        toks   = inst.get("tokens_expandits", [])
        teixit = inst.get("teixit", "?")
        n      = len(toks)
        if n == 0:
            continue

        n_voltes += 1
        teixits_set.add(teixit)
        n_inici[toks[0]]  += 1
        n_final[toks[-1]] += 1

        for i, tok in enumerate(toks):
            freq_token[tok] += 1
            pos = i / (n - 1) if n > 1 else 0.0
            positions_norm[tok].append(pos)
            freq_per_teixit[tok][teixit] += 1
            if i + 1 < n:
                bigram_counts[tok][toks[i + 1]] += 1

    return CorpusCounts(
        bigram_counts   = bigram_counts,
        positions_norm  = positions_norm,
        freq_per_teixit = freq_per_teixit,
        freq_token      = freq_token,
        n_inici         = n_inici,
        n_final         = n_final,
        n_voltes        = n_voltes,
        teixits_list    = sorted(teixits_set),
    )


# ─────────────────────────────────────────────
# MARGINALS PPMI
# ─────────────────────────────────────────────

@dataclass
class PPMIMarginals:
    total_bigrams: int
    left_sum:     np.ndarray   # Σ_b count(a,b)   (longitud V_full)
    right_sum:    np.ndarray   # Σ_a count(a,b)   (longitud V_full)
    left_alpha:   np.ndarray   # left_sum ** alpha
    right_alpha:  np.ndarray   # right_sum ** alpha
    z_left:       float
    z_right:      float
    alpha:        float


def compute_ppmi_marginals(
    bigram_counts: dict,
    tok_index_full: dict,
    alpha: float = 0.75,
) -> PPMIMarginals:
    """
    Calcula marginals esquerres i dretes (amb smoothing α a la distribució
    de context) per al càlcul PPMI smoothed segons Levy & Goldberg (2014).
    """
    V_full = len(tok_index_full)
    left_sum  = np.zeros(V_full)
    right_sum = np.zeros(V_full)
    total = 0

    for a, d in bigram_counts.items():
        if a not in tok_index_full:
            continue
        ai = tok_index_full[a]
        for b, c in d.items():
            if c <= 0 or b not in tok_index_full:
                continue
            bi = tok_index_full[b]
            left_sum[ai]  += c
            right_sum[bi] += c
            total += c

    left_alpha  = left_sum  ** alpha
    right_alpha = right_sum ** alpha
    z_left  = float(left_alpha.sum())  or 1.0
    z_right = float(right_alpha.sum()) or 1.0

    return PPMIMarginals(
        total_bigrams = total,
        left_sum      = left_sum,
        right_sum     = right_sum,
        left_alpha    = left_alpha,
        right_alpha   = right_alpha,
        z_left        = z_left,
        z_right       = z_right,
        alpha         = alpha,
    )


def ppmi_succ_vector(
    a: str,
    bigram_counts: dict,
    marg: PPMIMarginals,
    tok_index_full: dict,
) -> np.ndarray:
    """Vector PPMI de successors per al token a (longitud V_full)."""
    V_full = len(tok_index_full)
    v = np.zeros(V_full)
    if a not in tok_index_full or marg.total_bigrams == 0:
        return v
    ai = tok_index_full[a]
    if marg.left_sum[ai] == 0:
        return v
    p_a = marg.left_sum[ai] / marg.total_bigrams
    for b, c in bigram_counts.get(a, {}).items():
        if c <= 0 or b not in tok_index_full:
            continue
        bi = tok_index_full[b]
        p_ab = c / marg.total_bigrams
        p_b_alpha = marg.right_alpha[bi] / marg.z_right
        if p_a > 0 and p_b_alpha > 0:
            pmi = math.log2(p_ab / (p_a * p_b_alpha))
            v[bi] = max(0.0, pmi)
    return v


def ppmi_pred_vector(
    b: str,
    bigram_counts: dict,
    marg: PPMIMarginals,
    tok_index_full: dict,
) -> np.ndarray:
    """Vector PPMI de predecessors per al token b (longitud V_full)."""
    V_full = len(tok_index_full)
    v = np.zeros(V_full)
    if b not in tok_index_full or marg.total_bigrams == 0:
        return v
    bi = tok_index_full[b]
    if marg.right_sum[bi] == 0:
        return v
    p_b = marg.right_sum[bi] / marg.total_bigrams
    for a, d in bigram_counts.items():
        if a not in tok_index_full:
            continue
        ai = tok_index_full[a]
        c = d.get(b, 0)
        if c <= 0:
            continue
        p_ab = c / marg.total_bigrams
        p_a_alpha = marg.left_alpha[ai] / marg.z_left
        if p_b > 0 and p_a_alpha > 0:
            pmi = math.log2(p_ab / (p_a_alpha * p_b))
            v[ai] = max(0.0, pmi)
    return v


# ─────────────────────────────────────────────
# AUXILIARS ESTRUCTURALS
# ─────────────────────────────────────────────

def es_macro(tok: str) -> bool:
    return tok.startswith("(") and tok.endswith(")") and len(tok) >= 2


def profunditat_parens(tok: str) -> int:
    depth = max_d = 0
    for ch in tok:
        if ch == "(":
            depth += 1
            max_d = max(max_d, depth)
        elif ch == ")":
            depth -= 1
    return max_d


# ─────────────────────────────────────────────
# MATRIU DE CARACTERÍSTIQUES
# Layout (per fila):
#   [ succ_PPMI (V_full) | pred_PPMI (V_full) | posicional (6) |
#     estructural (3)   | per_teixit (T) ]
# ─────────────────────────────────────────────

FEATURE_DIM_FIXED = 6 + 3   # posicional + estructural


def feature_matrix_dim(V_full: int, T: int) -> int:
    return 2 * V_full + FEATURE_DIM_FIXED + T


def build_feature_matrix(
    vocab: list,                 # tokens clusteritzables (ordre fix)
    vocab_full: list,            # tot el vocab del corpus de referència
    tok_index_full: dict,
    counts: CorpusCounts,
    marg: PPMIMarginals,
    *,
    p_inici: dict,               # dict[token] -> P(t | inici_volta)
    p_final: dict,               # dict[token] -> P(t | final_volta)
    h_succ: dict,                # dict[token] -> H(successors)
    h_pred: dict,                # dict[token] -> H(predecessors)
) -> np.ndarray:
    """
    Construeix la matriu X (V × dim) amb el layout estàndard. Les estadístiques
    `p_inici`, `p_final`, `h_succ`, `h_pred` es proporcionen com a dicts:
      · Fase 3.2 les llegeix de markov_model.json (corpus complet).
      · Fase 3.2b les recalcula sobre cada mostra bootstrap.
    """
    V_full = len(vocab_full)
    T = len(counts.teixits_list)
    teixit_index = {t: i for i, t in enumerate(counts.teixits_list)}
    dim = feature_matrix_dim(V_full, T)

    V = len(vocab)
    X = np.zeros((V, dim), dtype=np.float64)

    o_pos  = 2 * V_full
    o_str  = 2 * V_full + 6
    o_teix = 2 * V_full + 6 + 3

    for i, tok in enumerate(vocab):
        # PPMI successors / predecessors
        X[i, 0:V_full]            = ppmi_succ_vector(tok, counts.bigram_counts, marg, tok_index_full)
        X[i, V_full:2 * V_full]   = ppmi_pred_vector(tok, counts.bigram_counts, marg, tok_index_full)

        # posicional
        poss = counts.positions_norm.get(tok, [])
        X[i, o_pos + 0] = float(p_inici.get(tok, 0.0))
        X[i, o_pos + 1] = float(p_final.get(tok, 0.0))
        X[i, o_pos + 2] = float(h_succ.get(tok, 0.0))
        X[i, o_pos + 3] = float(h_pred.get(tok, 0.0))
        X[i, o_pos + 4] = float(np.mean(poss)) if poss else 0.0
        X[i, o_pos + 5] = float(np.std(poss))  if poss else 0.0

        # estructural
        X[i, o_str + 0] = 1.0 if es_macro(tok) else 0.0
        X[i, o_str + 1] = float(profunditat_parens(tok))
        X[i, o_str + 2] = float(len(tok))

        # perfil per teixit (normalitzat)
        total = sum(counts.freq_per_teixit[tok].values())
        if total > 0:
            for tx, c in counts.freq_per_teixit[tok].items():
                X[i, o_teix + teixit_index[tx]] = c / total

    return X


def entropia_shannon_counts(counts_dict: dict) -> float:
    """Entropia de Shannon a partir d'un dict d'enters."""
    total = sum(counts_dict.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h
