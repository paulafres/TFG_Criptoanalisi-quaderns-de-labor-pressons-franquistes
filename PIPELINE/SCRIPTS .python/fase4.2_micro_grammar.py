# -*- coding: utf-8 -*-
"""
FASE 4.2 — Gramàtica MICRO: classes de token dins de volta
===========================================================

Construeix un model bigrama (P(Cx | Cy)) sobre seqüències de classes de
token (resultat de la Fase 3.2), una seqüència per volta. També calcula
entropia condicional global, surprisal per posició dins de cada volta i
agregats per volta (mean/max surprisal, perplexitat, transicions amb
probabilitat zero).

Input:
  res_fase2/instruction_dataset.json       (Fase 2)
  res_fase3.2/distributional_classes.json (Fase 3.2)

Output:
  res_fase4.2/micro_grammar.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase4_lib_ngram_grammar as ngm  # type: ignore

from config import (
    BASE, RES_FASE_4_2, ensure_dir,
    F2_INSTRUCTION_DATASET, F3_2_DISTRIB_CLASSES, F4_2_MICRO_GRAMMAR,
)

IN_DATASET = F2_INSTRUCTION_DATASET
IN_CLASSES = F3_2_DISTRIB_CLASSES
OUT_DIR = RES_FASE_4_2
OUT_JSON = F4_2_MICRO_GRAMMAR

NGRAM_ORDER = 2
SMOOTHING_K = 0.5


def main():
    ensure_dir(OUT_DIR)

    with open(IN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(IN_CLASSES, "r", encoding="utf-8") as f:
        classes_doc = json.load(f)

    token_to_class: dict[str, str] = {
        tok: info["classe"] for tok, info in classes_doc["tokens"].items()
    }
    class_labels: list[str] = sorted(
        classes_doc["classes"].keys(),
        key=lambda c: (0, int(c[1:])) if c[1:].isdigit() else (1, c),
    )

    # Traduir cada volta a seqüència de classes (saltant tokens desconeguts)
    seqs_per_volta: list[list[str]] = []
    meta_per_volta: list[dict] = []   # info per casar amb el sortida
    for v in dataset:
        seq_cls: list[str] = []
        n_unknown = 0
        for tok in v.get("tokens_expandits", []) or []:
            cls = token_to_class.get(tok)
            if cls is None:
                n_unknown += 1
                continue
            seq_cls.append(cls)
        seqs_per_volta.append(seq_cls)
        meta_per_volta.append({
            "teixit": v["teixit"],
            "volta": v["volta"],
            "longitud": len(seq_cls),
            "tokens_desconeguts": n_unknown,
        })

    # Models (smoothed i raw)
    model = ngm.build_ngram_model(
        seqs_per_volta, n=NGRAM_ORDER,
        alphabet=class_labels, smoothing_k=SMOOTHING_K,
    )
    model_raw = ngm.build_ngram_model(
        seqs_per_volta, n=NGRAM_ORDER,
        alphabet=class_labels, smoothing_k=0.0,
    )
    probs = ngm.transition_probabilities(model)
    ent = ngm.conditional_entropy(model)

    # Surprisal i resum per volta
    rows_per_volta: list[dict] = []
    for i, (seq, meta) in enumerate(zip(seqs_per_volta, meta_per_volta)):
        if len(seq) == 0:
            rows_per_volta.append({
                **meta,
                "mean_surprisal_bits": None,
                "max_surprisal_bits": None,
                "perplexity": None,
                "num_zero_prob_transitions": 0,
                "num_unseen_contexts": 0,
                "num_positions_scored": 0,
            })
            continue
        summary = ngm.sequence_anomaly_summary(
            seq, model, backoff_uniform=True
        )
        # Compte de transicions amb count=0 al model RAW (zero-prob "dur")
        n_raw_zero = _count_raw_zeros(seq, model_raw)
        rows_per_volta.append({
            **meta,
            **summary,
            "num_zero_prob_raw_transitions": n_raw_zero,
        })

    # Transicions rares (count <=2 al model raw, excloent inicis)
    rare_transitions = []
    for ctx, counts in model_raw["ngram_counts"].items():
        for sym, c in counts.items():
            if c <= 2 and ctx != ngm.BOS:
                rare_transitions.append({
                    "context": ctx, "symbol": sym, "count": c,
                })

    # ------------------ Escriure sortides ------------------
    out_doc = {
        "metadades": {
            "nivell": "micro (classe-Cx-dins-de-volta)",
            "ngram_order": NGRAM_ORDER,
            "smoothing_k": SMOOTHING_K,
            "alfabet": class_labels,
            "vocab_size": model["vocab_size"],
            "num_seqs": model["num_sequences"],
            "num_tokens": model["num_tokens"],
            "H_global_bits": ent["H_global_bits"],
            "n_contexts": ent["n_contexts"],
            "num_rare_transitions_count_le_2": len(rare_transitions),
        },
        "model": {
            "ngram_counts": model["ngram_counts"],
            "context_totals": model["context_totals"],
            "transition_probabilities": {
                ctx: {sym: round(p, 6) for sym, p in pdict.items()}
                for ctx, pdict in probs.items()
            },
            "H_per_context_bits": ent["H_per_context"],
        },
        "rare_transitions": rare_transitions,
        "per_volta": rows_per_volta,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    # ------------------ Resum a consola ------------------
    print("FASE 4.2 COMPLETADA (gramàtica MICRO)")
    print(f"  Voltes (seqüències)   : {model['num_sequences']}")
    print(f"  Classes Cx            : {model['vocab_size']}")
    print(f"  Tokens totals         : {model['num_tokens']}")
    print(f"  Contextos vistos      : {ent['n_contexts']}")
    print(f"  H_global              : {ent['H_global_bits']:.4f} bits")
    print(f"  Transicions rares (≤2): {len(rare_transitions)}")

    valid = [r for r in rows_per_volta if r["mean_surprisal_bits"] is not None]
    if valid:
        mu = sum(r["mean_surprisal_bits"] for r in valid) / len(valid)
        print(f"  Surprisal mitjà global: {mu:.3f} bits")

        # Top voltes amb surprisal mitjana més alta
        top_avg = sorted(valid, key=lambda r: r["mean_surprisal_bits"], reverse=True)[:5]
        print("\n  Top-5 voltes amb H̄ MICRO més alta:")
        for r in top_avg:
            print(f"    {r['teixit']:<22} v{r['volta']:<4} "
                  f"H̄={r['mean_surprisal_bits']:.2f}  perpl={r['perplexity']:.2f}  "
                  f"len={r['longitud']}  zero_raw={r.get('num_zero_prob_raw_transitions',0)}")

        # Top voltes amb més transicions zero-prob (raw)
        top_zero = sorted(valid, key=lambda r: r.get("num_zero_prob_raw_transitions", 0),
                          reverse=True)[:5]
        print("\n  Top-5 voltes amb més transicions count=0 (raw):")
        for r in top_zero:
            print(f"    {r['teixit']:<22} v{r['volta']:<4} "
                  f"zero_raw={r.get('num_zero_prob_raw_transitions',0)}  "
                  f"H̄={r['mean_surprisal_bits']:.2f}  len={r['longitud']}")

    print(f"\n  JSON : {OUT_JSON.relative_to(BASE.parent)}")


def _count_raw_zeros(seq: list[str], model_raw: dict) -> int:
    """Compta posicions on (context, símbol) té count=0 al model raw."""
    n = model_raw["n"]
    counts = model_raw["ngram_counts"]
    padded = [ngm.BOS] * (n - 1) + list(seq) + [ngm.EOS]
    z = 0
    for i in range(len(padded) - n + 1):
        ctx = "|".join(padded[i:i + n - 1])
        sym = padded[i + n - 1]
        if counts.get(ctx, {}).get(sym, 0) == 0:
            z += 1
    return z


if __name__ == "__main__":
    main()
