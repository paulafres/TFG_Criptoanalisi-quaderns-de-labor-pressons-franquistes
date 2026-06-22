# -*- coding: utf-8 -*-
"""
FASE 3.3 — Signatures distribucionals de cada volta
====================================================

Per a cada volta calcula:
  - composicio_classes (counts) i composicio_normalitzada (fraccions)
  - longitud_total, num_macros, prof_max, frac_outlier
  - dist_cosinus_volta_anterior (dins el mateix teixit)
  - z_salt_global  (Z-score del salt respecte a tots els salts del corpus)
  - dist_centroide_teixit (cosinus contra el centroide del seu teixit)
  - z_intra_teixit  (Z-score de la distància al centroide dins el seu teixit)

No s'assumeix res del significat dels tokens: només es fa servir la classe
distribucional emergent calculada a la Fase 3.2.
"""

from __future__ import annotations
import json
import math
import csv
from pathlib import Path
from collections import defaultdict

from config import (
    BASE, RES_FASE_3_3, ensure_dir,
    F2_INSTRUCTION_DATASET, F3_2_DISTRIB_CLASSES, F3_3_VOLTA_SIGNATURES,
)

# ---------------------------------------------------------------------------
# Configuració
# ---------------------------------------------------------------------------
INPUT_DATASET = F2_INSTRUCTION_DATASET
INPUT_CLASSES = F3_2_DISTRIB_CLASSES
OUT_DIR       = RES_FASE_3_3
OUT_JSON      = F3_3_VOLTA_SIGNATURES
OUT_CSV       = OUT_DIR / "volta_signatures.csv"
ensure_dir(OUT_DIR)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilitats vectorials
# ---------------------------------------------------------------------------
def cosine_distance(u: list[float], v: list[float]) -> float:
    """1 - cos(u,v). Retorna 1.0 si algun vector és nul."""
    num = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu < EPS or nv < EPS:
        return 1.0
    cos = num / (nu * nv)
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(var)


def zscore(x: float, mu: float, sd: float) -> float:
    if sd < EPS:
        return 0.0
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Càrrega
# ---------------------------------------------------------------------------
def load_inputs():
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    with open(INPUT_CLASSES, "r", encoding="utf-8") as f:
        classes_doc = json.load(f)

    token_to_class: dict[str, str] = {
        tok: info["classe"] for tok, info in classes_doc["tokens"].items()
    }
    token_outlier: dict[str, bool] = {
        tok: bool(info.get("es_outlier", False))
        for tok, info in classes_doc["tokens"].items()
    }
    class_labels: list[str] = sorted(
        classes_doc["classes"].keys(),
        key=lambda c: (0, int(c[1:])) if c[1:].isdigit() else (1, c),
    )
    return dataset, token_to_class, token_outlier, class_labels


# ---------------------------------------------------------------------------
# Càlcul de signatures
# ---------------------------------------------------------------------------
def compute_signature(volta_entry: dict,
                      token_to_class: dict[str, str],
                      token_outlier: dict[str, bool],
                      class_labels: list[str]) -> dict:
    tokens_exp = volta_entry.get("tokens_expandits", []) or []
    tokens_atom = volta_entry.get("tokens_atomics", []) or []

    composicio = {c: 0 for c in class_labels}
    n_outliers = 0
    n_unknown = 0
    for tok in tokens_exp:
        cls = token_to_class.get(tok)
        if cls is None:
            n_unknown += 1
            continue
        composicio[cls] += 1
        if token_outlier.get(tok, False):
            n_outliers += 1

    total = sum(composicio.values())
    if total > 0:
        composicio_norm = {c: composicio[c] / total for c in class_labels}
    else:
        composicio_norm = {c: 0.0 for c in class_labels}

    num_macros = sum(1 for t in tokens_atom if t.get("es_macro"))
    prof_max = max((t.get("profunditat", 0) for t in tokens_atom), default=0)
    frac_outlier = (n_outliers / total) if total > 0 else 0.0

    # ─── Blocs verticals (afegit per Tanda 3) ────────────────
    # Provenen de l'expansió B2 a Fase 2: cada `(seq-Nv)` queda
    # registrat com a metadada {pos, seq, n_voltes} sense ser
    # desplegat horitzontalment.
    blocks = volta_entry.get("blocks_verticals", []) or []
    num_blocks_v   = len(blocks)
    n_voltes_vals  = [b.get("n_voltes", 0) for b in blocks]
    seq_lens       = [len(str(b.get("seq", ""))) for b in blocks]
    sum_n_voltes_v = sum(n_voltes_vals)
    max_n_voltes_v = max(n_voltes_vals) if n_voltes_vals else 0
    max_seq_len_v  = max(seq_lens) if seq_lens else 0
    frac_verticals = (num_blocks_v / total) if total > 0 else 0.0

    return {
        "teixit": volta_entry["teixit"],
        "fitxer": volta_entry.get("fitxer"),
        "volta": volta_entry["volta"],
        "longitud_total": total,
        "num_tokens_atomics": len(tokens_atom),
        "num_macros": num_macros,
        "prof_max": prof_max,
        "frac_outlier": round(frac_outlier, 6),
        "tokens_desconeguts": n_unknown,
        "composicio_classes": composicio,
        "composicio_normalitzada": {c: round(v, 6) for c, v in composicio_norm.items()},
        # Features verticals
        "num_blocks_verticals":      num_blocks_v,
        "sum_n_voltes_verticals":    sum_n_voltes_v,
        "max_n_voltes_verticals":    max_n_voltes_v,
        "max_seq_len_vertical":      max_seq_len_v,
        "frac_verticals":            round(frac_verticals, 6),
    }


def vector_from_signature(sig: dict, class_labels: list[str]) -> list[float]:
    return [sig["composicio_normalitzada"][c] for c in class_labels]


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, t2c, t_out, class_labels = load_inputs()

    # 1) Signatures crues
    signatures: list[dict] = [
        compute_signature(v, t2c, t_out, class_labels) for v in dataset
    ]
    vectors: list[list[float]] = [
        vector_from_signature(s, class_labels) for s in signatures
    ]

    # 2) Distàncies entre voltes consecutives (dins el mateix teixit)
    salts: list[float] = []
    for i, sig in enumerate(signatures):
        if i == 0 or signatures[i - 1]["teixit"] != sig["teixit"]:
            sig["dist_cosinus_volta_anterior"] = None
        else:
            d = cosine_distance(vectors[i - 1], vectors[i])
            sig["dist_cosinus_volta_anterior"] = round(d, 6)
            salts.append(d)

    # 3) Z-score global de salts
    mu_g, sd_g = mean_std(salts)
    for sig in signatures:
        d = sig["dist_cosinus_volta_anterior"]
        sig["z_salt_global"] = (
            round(zscore(d, mu_g, sd_g), 6) if d is not None else None
        )

    # 4) Centroide per teixit + distància i Z-score intra-teixit
    teixit_idx: dict[str, list[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        teixit_idx[sig["teixit"]].append(i)

    centroides: dict[str, list[float]] = {}
    for teixit, idxs in teixit_idx.items():
        n = len(idxs)
        if n == 0:
            continue
        dim = len(vectors[idxs[0]])
        cent = [0.0] * dim
        for i in idxs:
            for k in range(dim):
                cent[k] += vectors[i][k]
        cent = [x / n for x in cent]
        centroides[teixit] = cent

    # distàncies al centroide
    dists_centroide_per_teixit: dict[str, list[float]] = defaultdict(list)
    dists_centroide: list[float | None] = [None] * len(signatures)
    for i, sig in enumerate(signatures):
        d = cosine_distance(vectors[i], centroides[sig["teixit"]])
        dists_centroide[i] = d
        dists_centroide_per_teixit[sig["teixit"]].append(d)

    # estadístics intra-teixit
    stats_teixit: dict[str, tuple[float, float]] = {
        t: mean_std(ds) for t, ds in dists_centroide_per_teixit.items()
    }

    for i, sig in enumerate(signatures):
        d = dists_centroide[i]
        mu_t, sd_t = stats_teixit[sig["teixit"]]
        sig["dist_centroide_teixit"] = round(d, 6)
        sig["z_intra_teixit"] = round(zscore(d, mu_t, sd_t), 6)

    # 5) Sortida JSON
    out_doc = {
        "metadades": {
            "num_voltes": len(signatures),
            "num_teixits": len(teixit_idx),
            "classes_usades": class_labels,
            "dim_signatura": len(class_labels),
            "salts_globals": {
                "n": len(salts),
                "mitjana": round(mu_g, 6),
                "desv_std": round(sd_g, 6),
            },
            "centroides_per_teixit": {
                t: [round(x, 6) for x in c] for t, c in centroides.items()
            },
        },
        "voltes": signatures,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    # 6) Sortida CSV
    csv_cols = [
        "teixit", "volta", "longitud_total", "num_tokens_atomics",
        "num_macros", "prof_max", "frac_outlier", "tokens_desconeguts",
        "num_blocks_verticals", "sum_n_voltes_verticals",
        "max_n_voltes_verticals", "max_seq_len_vertical", "frac_verticals",
        "dist_cosinus_volta_anterior", "z_salt_global",
        "dist_centroide_teixit", "z_intra_teixit",
    ] + [f"frac_{c}" for c in class_labels]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_cols)
        for sig in signatures:
            row = [
                sig["teixit"], sig["volta"], sig["longitud_total"],
                sig["num_tokens_atomics"], sig["num_macros"], sig["prof_max"],
                sig["frac_outlier"], sig["tokens_desconeguts"],
                sig["num_blocks_verticals"], sig["sum_n_voltes_verticals"],
                sig["max_n_voltes_verticals"], sig["max_seq_len_vertical"],
                sig["frac_verticals"],
                sig["dist_cosinus_volta_anterior"], sig["z_salt_global"],
                sig["dist_centroide_teixit"], sig["z_intra_teixit"],
            ]
            row += [sig["composicio_normalitzada"][c] for c in class_labels]
            w.writerow(row)

    # 7) Resum a consola
    print("FASE 3.3 COMPLETADA")
    print(f"  Voltes processades   : {len(signatures)}")
    print(f"  Teixits              : {len(teixit_idx)}")
    print(f"  Classes utilitzades  : {len(class_labels)}")
    print(f"  Salts calculats      : {len(salts)}")
    print(f"  Salt mitjà (cos.)    : {round(mu_g, 4)}  (sd={round(sd_g, 4)})")

    top_salts = sorted(
        [s for s in signatures if s["dist_cosinus_volta_anterior"] is not None],
        key=lambda s: s["dist_cosinus_volta_anterior"],
        reverse=True,
    )[:5]
    print("\n  Top-5 salts entre voltes consecutives:")
    for s in top_salts:
        print(f"    {s['teixit']:<22} v{s['volta']:<4} "
              f"d={s['dist_cosinus_volta_anterior']:.3f}  z={s['z_salt_global']:.2f}")

    top_centroide = sorted(
        signatures, key=lambda s: s["dist_centroide_teixit"], reverse=True
    )[:5]
    print("\n  Top-5 voltes més atípiques dins el seu teixit:")
    for s in top_centroide:
        print(f"    {s['teixit']:<22} v{s['volta']:<4} "
              f"d_cent={s['dist_centroide_teixit']:.3f}  z={s['z_intra_teixit']:.2f}")

    print(f"\n  JSON: {OUT_JSON.relative_to(BASE.parent)}")
    print(f"  CSV : {OUT_CSV.relative_to(BASE.parent)}")


if __name__ == "__main__":
    main()
