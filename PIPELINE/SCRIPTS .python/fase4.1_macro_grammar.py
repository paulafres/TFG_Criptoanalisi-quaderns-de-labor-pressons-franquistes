# -*- coding: utf-8 -*-
"""
FASE 4.1 — Gramàtica MACRO: tipus de volta dins de teixit
==========================================================

Construeix un model bigrama (P(VTx | VTy)) sobre les seqüències de tipus de
volta produïdes per la Fase 3.4. També calcula entropies condicionals i
surprisal per posició dins de cada teixit. Aquest és l'input "macro" que
la Fase 5 fusionarà amb el canal micro (Fase 4.2).

Input:
  res_fase3.4/sequencies_teixits.json   (Fase 3.4)

Output:
  res_fase4.1/macro_grammar.json
  res_fase4.1/macro_surprisal_per_volta.csv
"""

from __future__ import annotations
import json
import csv
import sys
from pathlib import Path

# Permet importar el helper que viu al mateix directori
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase4_lib_ngram_grammar as ngm  # type: ignore

from config import (
    BASE, RES_FASE_4_1, ensure_dir,
    F3_4_SEQUENCIES_TEIXITS, F4_1_MACRO_GRAMMAR, F4_1_MACRO_SURPRISAL_PER_VOLTA,
)

IN_SEQS = F3_4_SEQUENCIES_TEIXITS
OUT_DIR = RES_FASE_4_1
OUT_JSON = F4_1_MACRO_GRAMMAR
OUT_CSV_SURP = F4_1_MACRO_SURPRISAL_PER_VOLTA

NGRAM_ORDER = 2           # bigrama: les seqüències macro són curtes (17-131)
SMOOTHING_K = 0.5         # add-k per evitar 0s als reports


def main():
    ensure_dir(OUT_DIR)
    with open(IN_SEQS, "r", encoding="utf-8") as f:
        doc = json.load(f)

    alphabet = doc["metadades"]["alfabet"]
    teixits = doc["sequencies"]
    seqs_per_teixit: dict[str, list[str]] = {
        t: data["cadena"] for t, data in teixits.items()
    }
    voltes_per_teixit: dict[str, list[dict]] = {
        t: data["voltes"] for t, data in teixits.items()
    }
    all_seqs = list(seqs_per_teixit.values())

    # ------------------ Model global ------------------
    model_global = ngm.build_ngram_model(
        all_seqs, n=NGRAM_ORDER,
        alphabet=alphabet, smoothing_k=SMOOTHING_K,
    )
    probs_global = ngm.transition_probabilities(model_global)
    ent_global = ngm.conditional_entropy(model_global)

    # També un model "observacional" sense smoothing per detectar
    # transicions amb count=0 reals
    model_raw = ngm.build_ngram_model(
        all_seqs, n=NGRAM_ORDER,
        alphabet=alphabet, smoothing_k=0.0,
    )

    # ------------------ Surprisal per volta ------------------
    # Una "posició" = una transició entre dos tipus de volta consecutius
    surprisal_rows: list[dict] = []
    for teixit, seq in seqs_per_teixit.items():
        items = ngm.surprisal_per_position(seq, model_global, backoff_uniform=True)
        voltes = voltes_per_teixit[teixit]
        # items té len(seq)+1 transicions (incloent EOS). Casem amb les voltes:
        # items[i] descriu la transició cap a la posició i de la seqüència.
        for i, it in enumerate(items):
            if it["symbol"] == ngm.EOS:
                # transició final → no correspon a cap volta del corpus
                surprisal_rows.append({
                    "teixit": teixit,
                    "volta": None,
                    "tipus": ngm.EOS,
                    "context": "|".join(it["context"]),
                    "p": it["p"],
                    "surprisal_bits": it["surprisal_bits"],
                    "context_seen": it["context_seen"],
                    "is_zero_prob_raw": _is_raw_zero(it["context"], ngm.EOS, model_raw),
                })
            else:
                v = voltes[i]
                surprisal_rows.append({
                    "teixit": teixit,
                    "volta": v["volta"],
                    "tipus": v["tipus"],
                    "context": "|".join(it["context"]),
                    "p": it["p"],
                    "surprisal_bits": it["surprisal_bits"],
                    "context_seen": it["context_seen"],
                    "is_zero_prob_raw": _is_raw_zero(it["context"], v["tipus"], model_raw),
                })

    # Resum per teixit
    summary_per_teixit: dict[str, dict] = {}
    for teixit, seq in seqs_per_teixit.items():
        summary_per_teixit[teixit] = ngm.sequence_anomaly_summary(
            seq, model_global, backoff_uniform=True
        )

    # Transicions impossibles (count=0 al corpus) observades dins el corpus mateix
    # → no haurien d'aparèixer (per definició no n'hi ha si entrenem amb totes
    # les seqüències). Però guardem la llista de transicions amb count <=1 com
    # a "rares".
    rare_transitions = []
    for ctx, counts in model_raw["ngram_counts"].items():
        for sym, c in counts.items():
            if c == 1 and ctx != ngm.BOS:  # exclou inicis (sovint únics)
                rare_transitions.append({
                    "context": ctx, "symbol": sym, "count": c,
                })

    # ------------------ Escriure sortides ------------------
    out_doc = {
        "metadades": {
            "nivell": "macro (volta-en-teixit)",
            "ngram_order": NGRAM_ORDER,
            "smoothing_k": SMOOTHING_K,
            "alfabet": alphabet,
            "vocab_size": model_global["vocab_size"],
            "num_seqs": model_global["num_sequences"],
            "num_tokens": model_global["num_tokens"],
            "H_global_bits": ent_global["H_global_bits"],
            "n_contexts": ent_global["n_contexts"],
            "num_rare_transitions_count1": len(rare_transitions),
        },
        "model": {
            "ngram_counts": model_global["ngram_counts"],
            "context_totals": model_global["context_totals"],
            "transition_probabilities": {
                ctx: {sym: round(p, 6) for sym, p in pdict.items()}
                for ctx, pdict in probs_global.items()
            },
            "H_per_context_bits": ent_global["H_per_context"],
        },
        "summary_per_teixit": summary_per_teixit,
        "rare_transitions_count1": rare_transitions,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    # CSV de surprisal per volta
    with open(OUT_CSV_SURP, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teixit", "volta", "tipus", "context",
                    "p", "surprisal_bits", "context_seen", "is_zero_prob_raw"])
        for r in surprisal_rows:
            w.writerow([r["teixit"], r["volta"], r["tipus"], r["context"],
                        r["p"], r["surprisal_bits"],
                        int(r["context_seen"]), int(r["is_zero_prob_raw"])])

    # ------------------ Resum a consola ------------------
    print("FASE 4.1 COMPLETADA (gramàtica MACRO)")
    print(f"  Seqüències (teixits)  : {model_global['num_sequences']}")
    print(f"  Tipus de volta (V|E)  : {model_global['vocab_size']}")
    print(f"  Contextos vistos      : {ent_global['n_contexts']}")
    print(f"  H_global              : {ent_global['H_global_bits']:.4f} bits")
    print(f"  Transicions rares (=1): {len(rare_transitions)}")

    print("\n  Surprisal mitjà per teixit (bits/transició):")
    items_sorted = sorted(summary_per_teixit.items(),
                          key=lambda kv: kv[1]["mean_surprisal_bits"],
                          reverse=True)
    for t, s in items_sorted:
        print(f"    {t:<22}  H̄={s['mean_surprisal_bits']:.2f}  "
              f"perpl={s['perplexity']:.2f}  "
              f"len={s['len']}  zero={s['num_zero_prob_transitions']}")

    # Top-5 voltes amb surprisal màxima
    top = sorted(
        [r for r in surprisal_rows
         if r["surprisal_bits"] is not None and r["volta"] is not None],
        key=lambda r: r["surprisal_bits"], reverse=True,
    )[:5]
    print("\n  Top-5 voltes amb surprisal MACRO més alta:")
    for r in top:
        print(f"    {r['teixit']:<22} v{r['volta']:<4} "
              f"{r['context']}→{r['tipus']}  "
              f"surp={r['surprisal_bits']:.2f}  p={r['p']:.4f}")

    print(f"\n  JSON : {OUT_JSON.relative_to(BASE.parent)}")
    print(f"  CSV  : {OUT_CSV_SURP.relative_to(BASE.parent)}")


def _is_raw_zero(context: list[str], symbol: str, model_raw: dict) -> bool:
    ctx_key = "|".join(context)
    counts = model_raw["ngram_counts"].get(ctx_key, {})
    return counts.get(symbol, 0) == 0


if __name__ == "__main__":
    main()
