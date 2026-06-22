# -*- coding: utf-8 -*-
"""
FASE 5.2 — DETECCIÓ DE SEGMENTS ANÒMALS (CANDIDATS A ESTEGANOGRAFIA)
=====================================================================

Hipòtesi de treball
-------------------
Si Carmen Machado va amagar un missatge dins d'un teixit, és més probable
que ocupi un TROS CONTIGU de voltes (un fragment de text) que no pas
voltes aïllades disperses. Aquesta fase busca SEGMENTS de voltes
consecutives anòmales dins de cada teixit.

Mètode
------
0. SCORE COMBINAT: per a cada volta unim DOS canals d'anomalia:
   · gramatical → `score_anomalia` (Fase 5.1)
   · lèxic      → `prioritat` (Fase 5.1b = z_runs_abs + 10·n_rare; 0 si
                  la volta no és candidata lèxica)
   Estandarditzem cada canal a z-score sobre tot el corpus i els sumem
   ponderats: score_combinat = W_GRAM·z(gram) + W_LEX·z(lèxic).
   Així una zona lèxicament densa (p. ex. Primavera/Helga) també pot
   generar segment, encara que la gramàtica hi sigui neta.
1. Per a cada teixit, ordenem les voltes per `volta` i obtenim la sèrie
   `score_combinat`.
2. Threshold global τ = percentil 60 dels scores combinats del corpus.
   Una volta és "alta" si score_combinat >= τ.
3. Detectem RUNS màxims de voltes consecutives "altes". Un segment
   candidat és un run de longitud >= MIN_SEG_LEN (= 3).
4. Test de permutació (1000 iteracions) per a cada segment:
       H0 = l'ordre dels scores dins del teixit és intercanviable.
       Estadístic = mitjana dels scores del segment.
       p-valor = fracció de permutacions on existeix una finestra de la
                 mateixa longitud amb mitjana >= observada.
5. Correcció FDR-Benjamini-Hochberg (α=0.05) sobre tots els p-valors.
6. Per a cada segment significatiu (q <= 0.05) generem el detall:
   - votes (nº voltes), score acumulat i mig
   - distribució de categories (CRÍTICA/ALTA/MODERADA/normal)
   - cross_canal: nº voltes amb senyal en els dos canals (micro+macro)

Sortida
-------
- res_fase5.2/anomaly_segments.json   (objectes amb tot el detall)
- res_fase5.2/anomaly_segments.csv    (taula plana per inspecció ràpida)
- res_fase5.2/anomaly_segments_report.txt  (resum llegible)
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from config import (
    F5_1_ANOMALY_REPORT_CSV,
    F5_1B_CRYPTO_VALIDATION,
    RES_FASE_5_2,
    F5_2_SEGMENTS_JSON,
    F5_2_SEGMENTS_CSV,
    F5_2_REPORT_TXT,
    RANDOM_STATE,
    ensure_dir,
)


# ───────────────────────── PARÀMETRES ────────────────────────────────────
THRESHOLD_PERCENTILE = 60     # percentil global de score per marcar volta "alta"
MIN_SEG_LEN          = 3      # un segment ha de tenir >= 3 voltes consecutives
N_PERM               = 1000   # iteracions del test de permutació
ALPHA_FDR            = 0.05   # tall del q-valor (FDR-BH)
W_GRAM               = 1.0    # pes del canal gramatical (5.1) al score combinat
W_LEX                = 1.0    # pes del canal lèxic (5.1b) al score combinat


# ───────────────────────── CÀRREGA ───────────────────────────────────────
def carrega_anomalies() -> list[dict]:
    rows: list[dict] = []
    with open(F5_1_ANOMALY_REPORT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["volta"] = int(row["volta"])
            row["score_anomalia"] = float(row["score_anomalia"])
            row["cross_canal"] = (row["cross_canal"].strip().lower() == "true")
            rows.append(row)
    return rows


def carrega_lexic() -> dict[tuple[str, int], float]:
    """Magnitud lèxica per volta des de 5.1b.

    Fa servir `prioritat` (= z_runs_abs + 10·n_rare). Les voltes que no
    apareixen als candidats de 5.1b tenen senyal lèxic 0."""
    out: dict[tuple[str, int], float] = {}
    if not F5_1B_CRYPTO_VALIDATION.exists():
        return out
    with open(F5_1B_CRYPTO_VALIDATION, encoding="utf-8") as f:
        doc = json.load(f)
    for c in doc.get("candidats_steganografia", []):
        try:
            key = (c["teixit"], int(c["volta"]))
        except (KeyError, TypeError, ValueError):
            continue
        val = c.get("prioritat")
        if val is None:
            val = (abs(c.get("z_runs_abs", 0.0) or 0.0)
                   + 10.0 * (c.get("n_rare", 0) or 0))
        out[key] = float(val)
    return out


# ───────────────────────── DETECCIÓ DE RUNS ──────────────────────────────
def maximal_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Retorna [(start_idx, end_idx_exclusive), ...] dels runs True
    de longitud >= min_len."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def max_window_mean(scores: np.ndarray, L: int) -> float:
    """Màxima mitjana sobre qualsevol finestra contigua de longitud L."""
    if L > len(scores):
        return float("-inf")
    csum = np.concatenate(([0.0], np.cumsum(scores)))
    sums = csum[L:] - csum[:-L]
    return float(sums.max() / L)


# ───────────────────────── PERMUTATION TEST ──────────────────────────────
def permutation_pvalue(scores: np.ndarray, L: int, observed_mean: float,
                       n_perm: int, rng: np.random.Generator) -> float:
    """p-valor 1-sided: P( max_window_mean(perm_scores, L) >= observed_mean ).

    Ús de regla "add-one" per evitar p=0 exacte.
    """
    if L >= len(scores):
        return 1.0
    arr = scores.copy()
    geq = 0
    for _ in range(n_perm):
        rng.shuffle(arr)
        if max_window_mean(arr, L) >= observed_mean - 1e-12:
            geq += 1
    return (geq + 1) / (n_perm + 1)


# ───────────────────────── FDR Benjamini-Hochberg ────────────────────────
def fdr_bh(pvals: list[float]) -> list[float]:
    """Q-valors (BH) corresponents a la llista de p-valors. Ordre conservat."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    qvals = [0.0] * n
    prev = 1.0
    # recorregut invers garantint monotonia decreixent
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        q = pvals[i] * n / (rank + 1)
        if q > 1.0:
            q = 1.0
        if q > prev:
            q = prev
        qvals[i] = q
        prev = q
    return qvals


# ───────────────────────── DETECCIÓ PRINCIPAL ────────────────────────────
def detecta_segments(rows: list[dict],
                     lexic: dict[tuple[str, int], float]) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)

    # ── Score combinat: z(gramatical 5.1) + z(lèxic 5.1b) ──
    g = np.array([r["score_anomalia"] for r in rows], dtype=float)
    l = np.array([lexic.get((r["teixit"], r["volta"]), 0.0) for r in rows],
                 dtype=float)

    def zstd(a: np.ndarray) -> np.ndarray:
        sd = a.std()
        return (a - a.mean()) / sd if sd > 1e-12 else a - a.mean()

    combinat = W_GRAM * zstd(g) + W_LEX * zstd(l)
    for r, c, lv in zip(rows, combinat, l):
        r["score_combinat"] = float(c)
        r["score_lexic"] = float(lv)

    all_scores = combinat
    threshold = float(np.percentile(all_scores, THRESHOLD_PERCENTILE))

    # Agrupem per teixit
    per_teixit: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_teixit[r["teixit"]].append(r)
    for t in per_teixit:
        per_teixit[t].sort(key=lambda x: x["volta"])

    candidates: list[dict] = []  # abans de FDR
    for teixit, recs in per_teixit.items():
        scores = np.array([r["score_combinat"] for r in recs], dtype=float)
        voltes = [r["volta"] for r in recs]
        mask = scores >= threshold

        runs = maximal_runs(mask, MIN_SEG_LEN)
        for (i0, i1) in runs:
            seg_scores = scores[i0:i1]
            L = i1 - i0
            obs_mean = float(seg_scores.mean())
            obs_sum  = float(seg_scores.sum())

            pval = permutation_pvalue(scores, L, obs_mean, N_PERM, rng)

            # Categories i cross-canal dins el segment
            seg_recs = recs[i0:i1]
            cat_counter = Counter(r["categoria"] for r in seg_recs)
            n_cross = sum(1 for r in seg_recs if r["cross_canal"])

            candidates.append({
                "teixit": teixit,
                "volta_inici": voltes[i0],
                "volta_final": voltes[i1 - 1],
                "n_voltes": L,
                "score_mitja": round(obs_mean, 4),
                "score_acumulat": round(obs_sum, 4),
                "p_valor": pval,
                "n_voltes_cross_canal": n_cross,
                "n_critiques": cat_counter.get("CRÍTICA", 0),
                "n_altes":     cat_counter.get("ALTA", 0),
                "n_moderades": cat_counter.get("MODERADA", 0),
                "n_normals":   cat_counter.get("normal", 0),
                "voltes_detall": [
                    {
                        "volta": r["volta"],
                        "tipus_volta": r["tipus_volta"],
                        "score_anomalia": round(r["score_anomalia"], 4),
                        "score_lexic": round(r["score_lexic"], 4),
                        "score_combinat": round(r["score_combinat"], 4),
                        "categoria": r["categoria"],
                        "cross_canal": r["cross_canal"],
                        "longitud_micro": int(r["longitud_micro"]),
                    }
                    for r in seg_recs
                ],
            })

    # Correcció FDR
    pvals = [c["p_valor"] for c in candidates]
    qvals = fdr_bh(pvals)
    for c, q in zip(candidates, qvals):
        c["q_valor"] = q
        c["significatiu_FDR"] = bool(q <= ALPHA_FDR)

    # Ordenació final: significatius primer, després per score acumulat
    candidates.sort(
        key=lambda c: (not c["significatiu_FDR"], -c["score_acumulat"])
    )

    return {
        "metadades": {
            "score": "combinat: W_GRAM·z(gramatical 5.1) + W_LEX·z(lèxic 5.1b)",
            "pes_gramatical": W_GRAM,
            "pes_lexic": W_LEX,
            "n_voltes_amb_senyal_lexic": int((l > 0).sum()),
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "threshold_score": round(threshold, 4),
            "min_seg_len": MIN_SEG_LEN,
            "n_perm": N_PERM,
            "alpha_fdr": ALPHA_FDR,
            "random_state": RANDOM_STATE,
            "n_voltes_corpus": len(rows),
            "n_segments_candidats": len(candidates),
            "n_segments_significatius": sum(
                1 for c in candidates if c["significatiu_FDR"]
            ),
        },
        "segments": candidates,
    }


# ───────────────────────── ESCRIPTURA ────────────────────────────────────
CSV_COLS = [
    "teixit", "volta_inici", "volta_final", "n_voltes",
    "score_mitja", "score_acumulat",
    "p_valor", "q_valor", "significatiu_FDR",
    "n_voltes_cross_canal",
    "n_critiques", "n_altes", "n_moderades", "n_normals",
]


def escriu_csv(doc: dict) -> None:
    with open(F5_2_SEGMENTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for s in doc["segments"]:
            w.writerow([
                s["teixit"],
                s["volta_inici"],
                s["volta_final"],
                s["n_voltes"],
                f"{s['score_mitja']:.4f}",
                f"{s['score_acumulat']:.4f}",
                f"{s['p_valor']:.4f}",
                f"{s['q_valor']:.4f}",
                "TRUE" if s["significatiu_FDR"] else "FALSE",
                s["n_voltes_cross_canal"],
                s["n_critiques"],
                s["n_altes"],
                s["n_moderades"],
                s["n_normals"],
            ])


def escriu_report(doc: dict) -> None:
    md = doc["metadades"]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" FASE 5.2 — SEGMENTS ANÒMALS CONTIGUS  (candidats a esteganografia)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Paràmetres:")
    lines.append(f"  · score = {md.get('pes_gramatical', 1)}·z(gramatical 5.1) + "
                 f"{md.get('pes_lexic', 1)}·z(lèxic 5.1b)")
    lines.append(f"  · voltes amb senyal lèxic (5.1b)        = "
                 f"{md.get('n_voltes_amb_senyal_lexic', '?')}")
    lines.append(f"  · llindar de score combinat (perc. {md['threshold_percentile']}) = "
                 f"{md['threshold_score']}")
    lines.append(f"  · longitud mínima de segment            = {md['min_seg_len']} voltes")
    lines.append(f"  · iteracions de permutació              = {md['n_perm']}")
    lines.append(f"  · α FDR (Benjamini-Hochberg)            = {md['alpha_fdr']}")
    lines.append(f"  · random_state                          = {md['random_state']}")
    lines.append("")
    lines.append(f"Voltes al corpus            : {md['n_voltes_corpus']}")
    lines.append(f"Segments candidats trobats  : {md['n_segments_candidats']}")
    lines.append(f"Segments significatius (q≤α): {md['n_segments_significatius']}")
    lines.append("")
    lines.append("-" * 72)
    lines.append(" RÀNQUING DE SEGMENTS")
    lines.append("-" * 72)
    lines.append(f"{'#':>3} {'teixit':<22} {'voltes':<14} {'L':>3} "
                 f"{'mean':>6} {'sum':>7} {'p':>6} {'q':>6} {'sig':>4} "
                 f"{'cx':>3} {'CRI':>3} {'ALT':>3} {'MOD':>3}")
    for i, s in enumerate(doc["segments"], 1):
        rng_str = f"v{s['volta_inici']}-v{s['volta_final']}"
        lines.append(
            f"{i:>3} {s['teixit'][:22]:<22} {rng_str:<14} {s['n_voltes']:>3} "
            f"{s['score_mitja']:>6.2f} {s['score_acumulat']:>7.2f} "
            f"{s['p_valor']:>6.3f} {s['q_valor']:>6.3f} "
            f"{'YES' if s['significatiu_FDR'] else ' no':>4} "
            f"{s['n_voltes_cross_canal']:>3} "
            f"{s['n_critiques']:>3} {s['n_altes']:>3} {s['n_moderades']:>3}"
        )

    lines.append("")
    lines.append("-" * 72)
    lines.append(" DETALL DELS SEGMENTS SIGNIFICATIUS")
    lines.append("-" * 72)
    sigs = [s for s in doc["segments"] if s["significatiu_FDR"]]
    if not sigs:
        lines.append("  (cap segment supera el llindar FDR)")
    for k, s in enumerate(sigs, 1):
        lines.append("")
        lines.append(f" [{k}] {s['teixit']}  ·  voltes "
                     f"v{s['volta_inici']}-v{s['volta_final']}  "
                     f"(L={s['n_voltes']}, q={s['q_valor']:.3f})")
        for d in s["voltes_detall"]:
            lines.append(
                f"     v{d['volta']:<4} {d['tipus_volta']:<4} "
                f"comb={d.get('score_combinat', 0.0):>6.2f} "
                f"(g={d['score_anomalia']:>5.2f} l={d.get('score_lexic', 0.0):>5.1f}) "
                f"cat={d['categoria']:<8} "
                f"cx={'Y' if d['cross_canal'] else 'n'}"
            )
    lines.append("")

    F5_2_REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


# ───────────────────────── MAIN ──────────────────────────────────────────
def main() -> None:
    ensure_dir(RES_FASE_5_2)
    print("== FASE 5.2 · Detecció de segments anòmals ==")
    rows = carrega_anomalies()
    lexic = carrega_lexic()
    print(f"   Voltes carregades: {len(rows)}  "
          f"(amb senyal lèxic 5.1b: {len(lexic)})")

    doc = detecta_segments(rows, lexic)
    md = doc["metadades"]
    print(f"   Llindar score combinat (P{md['threshold_percentile']}) = "
          f"{md['threshold_score']}")
    print(f"   Segments candidats     : {md['n_segments_candidats']}")
    print(f"   Segments significatius : {md['n_segments_significatius']} "
          f"(FDR α={md['alpha_fdr']})")

    F5_2_SEGMENTS_JSON.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    escriu_csv(doc)
    escriu_report(doc)

    print()
    print(f"   JSON   → {F5_2_SEGMENTS_JSON.name}")
    print(f"   CSV    → {F5_2_SEGMENTS_CSV.name}")
    print(f"   REPORT → {F5_2_REPORT_TXT.name}")
    print("== FASE 5.2 COMPLETADA ==")


if __name__ == "__main__":
    main()
