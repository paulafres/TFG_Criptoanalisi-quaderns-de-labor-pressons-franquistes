# -*- coding: utf-8 -*-
"""
lib_ngram_grammar.py
====================

Mòdul comú de gramàtica distribucional (n-grames + entropia condicional +
puntuació d'anomalia per posició). Reutilitzat per la Fase 4 (macro: tipus
de volta dins de teixit) i la Fase 4.5 (micro: tokens/classes dins de volta).

No fa cap suposició sobre la naturalesa dels símbols: només tracta amb
seqüències d'identificadors (`str`).

Markers especials utilitzats:
  BOS = "<BOS>"   inici de seqüència
  EOS = "<EOS>"   final de seqüència
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Iterable, Sequence

BOS = "<BOS>"
EOS = "<EOS>"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Construcció del model n-grama
# ---------------------------------------------------------------------------
def _wrap(seq: Sequence[str], n: int) -> list[str]:
    """Afegeix (n-1) marcadors BOS al davant i 1 EOS al final."""
    return [BOS] * (n - 1) + list(seq) + [EOS]


def build_ngram_model(
    sequences: Iterable[Sequence[str]],
    n: int = 2,
    alphabet: list[str] | None = None,
    smoothing_k: float = 0.0,
) -> dict:
    """
    Construeix un model n-grama (n=1,2,3).

    Retorna un diccionari amb:
      - "n"                 : ordre del model
      - "alphabet"          : símbols vistos (sense BOS/EOS) ordenats
      - "vocab_size"        : |alfabet| (sense BOS/EOS)
      - "num_sequences"     : nombre de seqüències processades
      - "num_tokens"        : suma de longituds (sense marcadors)
      - "ngram_counts"      : {context_str -> {symbol -> count}}
                              context_str = "|".join(context_tuple)
      - "context_totals"    : {context_str -> total}
      - "smoothing_k"       : valor d'add-k aplicat
      - "uses_BOS_EOS"      : True
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    seqs = [list(s) for s in sequences]
    seen_symbols: set[str] = set()
    for s in seqs:
        seen_symbols.update(s)

    if alphabet is None:
        alphabet = sorted(seen_symbols)
    else:
        # Garantir que els símbols vistos hi siguin
        missing = seen_symbols - set(alphabet)
        if missing:
            alphabet = sorted(set(alphabet) | missing)

    ngram_counts: dict[str, Counter] = defaultdict(Counter)
    context_totals: Counter = Counter()
    num_tokens = 0

    for s in seqs:
        num_tokens += len(s)
        padded = _wrap(s, n)
        for i in range(len(padded) - n + 1):
            context = tuple(padded[i:i + n - 1])
            symbol = padded[i + n - 1]
            ctx_key = "|".join(context)
            ngram_counts[ctx_key][symbol] += 1
            context_totals[ctx_key] += 1

    return {
        "n": n,
        "alphabet": alphabet,
        "vocab_size": len(alphabet),
        "num_sequences": len(seqs),
        "num_tokens": num_tokens,
        "ngram_counts": {ctx: dict(c) for ctx, c in ngram_counts.items()},
        "context_totals": dict(context_totals),
        "smoothing_k": smoothing_k,
        "uses_BOS_EOS": True,
    }


# ---------------------------------------------------------------------------
# Probabilitats i entropies
# ---------------------------------------------------------------------------
def transition_probabilities(model: dict) -> dict[str, dict[str, float]]:
    """
    P(symbol | context) per a cada context observat, amb add-k smoothing
    opcional sobre tot l'alfabet + EOS.
    """
    k = model["smoothing_k"]
    alphabet = model["alphabet"]
    # els símbols possibles després d'un context inclouen EOS, mai BOS
    targets = alphabet + [EOS]
    V = len(targets)

    probs: dict[str, dict[str, float]] = {}
    for ctx, counts in model["ngram_counts"].items():
        total = model["context_totals"][ctx]
        if k > 0:
            denom = total + k * V
            ctx_probs = {sym: (counts.get(sym, 0) + k) / denom for sym in targets}
        else:
            ctx_probs = {sym: counts.get(sym, 0) / total for sym in targets if counts.get(sym, 0) > 0}
        probs[ctx] = ctx_probs
    return probs


def conditional_entropy(model: dict) -> dict:
    """
    Entropia condicional H(X | context) per cada context observat,
    i entropia condicional global ponderada pel pes del context.
    Utilitza log base 2 (bits).
    """
    probs = transition_probabilities(model)
    h_per_ctx: dict[str, float] = {}
    total_count = sum(model["context_totals"].values())
    h_global = 0.0
    for ctx, p in probs.items():
        h = 0.0
        for _, pv in p.items():
            if pv > EPS:
                h -= pv * math.log2(pv)
        h_per_ctx[ctx] = h
        w = model["context_totals"][ctx] / total_count if total_count > 0 else 0.0
        h_global += w * h
    return {
        "H_per_context": {ctx: round(h, 4) for ctx, h in h_per_ctx.items()},
        "H_global_bits": round(h_global, 4),
        "n_contexts": len(h_per_ctx),
    }


# ---------------------------------------------------------------------------
# Surprise per posició (anomalia local)
# ---------------------------------------------------------------------------
def surprisal_per_position(
    sequence: Sequence[str],
    model: dict,
    backoff_uniform: bool = True,
) -> list[dict]:
    """
    Per cada posició i de la seqüència, calcula:
      - context (n-1 tokens previs, possiblement amb BOS)
      - símbol observat
      - P(símbol | context)         (segons el model, amb smoothing)
      - surprisal = -log2(P)
      - context_seen (bool)
      - is_zero_prob (bool)         True si P=0 amb el model sense smoothing
                                    (transició no observada al corpus)

    Si el context no s'ha vist mai:
      - Si backoff_uniform=True: assigna P = 1/(V+1) (alfabet + EOS)
      - Si backoff_uniform=False: marca P=0 i surprisal=inf
    """
    n = model["n"]
    counts = model["ngram_counts"]
    totals = model["context_totals"]
    k = model["smoothing_k"]
    V_plus = len(model["alphabet"]) + 1  # +1 per EOS
    padded = _wrap(sequence, n)
    out: list[dict] = []

    for i in range(len(padded) - n + 1):
        context = tuple(padded[i:i + n - 1])
        symbol = padded[i + n - 1]
        if symbol == EOS:
            # encara registrem la transició final, és informativa
            pass
        ctx_key = "|".join(context)
        ctx_total = totals.get(ctx_key, 0)
        sym_count = counts.get(ctx_key, {}).get(symbol, 0)

        if ctx_total == 0:
            # context no vist
            if backoff_uniform:
                p = 1.0 / V_plus
            else:
                p = 0.0
            context_seen = False
            is_zero_prob = True
        else:
            context_seen = True
            if k > 0:
                p = (sym_count + k) / (ctx_total + k * V_plus)
            else:
                p = sym_count / ctx_total
            is_zero_prob = (sym_count == 0)

        if p > EPS:
            surprisal = -math.log2(p)
        else:
            surprisal = float("inf")

        out.append({
            "position": i,
            "context": list(context),
            "symbol": symbol,
            "p": round(p, 8) if p < 1 else 1.0,
            "surprisal_bits": round(surprisal, 4) if surprisal != float("inf") else None,
            "context_seen": context_seen,
            "is_zero_prob": is_zero_prob,
        })
    return out


# ---------------------------------------------------------------------------
# Agregats útils per a reports / Fase 5
# ---------------------------------------------------------------------------
def sequence_anomaly_summary(
    sequence: Sequence[str],
    model: dict,
    backoff_uniform: bool = True,
) -> dict:
    """
    Resum d'anomalia d'una seqüència completa:
      - mean_surprisal, max_surprisal
      - num_zero_prob_transitions
      - num_unseen_contexts
      - perplexity (2^H_mitja, ignorant infinits si en queden)
    """
    items = surprisal_per_position(sequence, model, backoff_uniform=backoff_uniform)
    surps = [it["surprisal_bits"] for it in items if it["surprisal_bits"] is not None]
    n_zero = sum(1 for it in items if it["is_zero_prob"])
    n_unseen_ctx = sum(1 for it in items if not it["context_seen"])
    mean_s = sum(surps) / len(surps) if surps else 0.0
    max_s = max(surps) if surps else 0.0
    perplexity = 2 ** mean_s if surps else float("inf")
    return {
        "len": len(sequence),
        "mean_surprisal_bits": round(mean_s, 4),
        "max_surprisal_bits": round(max_s, 4),
        "perplexity": round(perplexity, 4),
        "num_zero_prob_transitions": n_zero,
        "num_unseen_contexts": n_unseen_ctx,
        "num_positions_scored": len(items),
    }


# ---------------------------------------------------------------------------
# Transicions impossibles vistes "fora del corpus" (oposat: les que el model
# considera nul·les). Útil per a la Fase 5 si volem llindar manualment.
# ---------------------------------------------------------------------------
def zero_probability_transitions(
    sequences: Iterable[Sequence[str]],
    model: dict,
) -> list[dict]:
    """
    Recorre les seqüències i retorna totes les posicions on s'observa una
    transició amb count=0 sota el model donat. Utilitzat per la Fase 5
    com a llista d'anomalies "dures".
    """
    out = []
    for idx, seq in enumerate(sequences):
        items = surprisal_per_position(seq, model, backoff_uniform=True)
        for it in items:
            if it["is_zero_prob"]:
                out.append({
                    "seq_index": idx,
                    "position": it["position"],
                    "context": it["context"],
                    "symbol": it["symbol"],
                    "context_seen": it["context_seen"],
                })
    return out
