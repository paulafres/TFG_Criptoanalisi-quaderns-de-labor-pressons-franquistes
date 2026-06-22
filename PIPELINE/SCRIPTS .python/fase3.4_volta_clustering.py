# -*- coding: utf-8 -*-
"""
FASE 3.4 — Clustering distribucional de voltes (tipus de volta emergents)
==========================================================================

Aplica la mateixa metodologia de la Fase 3.2 però al nivell superior:
  - Token  -> classe Cx  (Fase 3.2)
  - Volta  -> tipus VTx  (Fase 3.4)  <-- aquí

Input  : volta_signatures.json  (vector 12-dim per volta = composicio_normalitzada)
Procés : L2-normalització, KMeans amb cerca de k òptim per silhouette,
         detecció d'outliers (z>2 respecte al centroide del clúster).
Output :
  - volta_classes.json     : metadades + perfil de cada tipus VTx
  - volta_a_tipus.csv      : (teixit, volta, tipus, outlier_score, es_outlier)
  - sequencies_teixits.json: per cada teixit, la cadena de tipus de volta
                             (input directe per a la Fase 4 — gramàtica de voltes)
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    BASE, RES_FASE_3_4, ensure_dir, RANDOM_STATE,
    F3_3_VOLTA_SIGNATURES, F3_4_VOLTA_CLASSES, F3_4_SEQUENCIES_TEIXITS,
)

# ---------------------------------------------------------------------------
# Configuració
# ---------------------------------------------------------------------------
INPUT_SIGS = F3_3_VOLTA_SIGNATURES
OUT_DIR    = RES_FASE_3_4
OUT_JSON   = F3_4_VOLTA_CLASSES
OUT_CSV    = OUT_DIR / "volta_a_tipus.csv"
OUT_SEQ    = F3_4_SEQUENCIES_TEIXITS

K_MIN, K_MAX = 2, 18
N_INIT = 20
ensure_dir(OUT_DIR)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilitats
# ---------------------------------------------------------------------------
def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return X / norms


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels)) < 2:
        return -1.0
    return float(silhouette_score(X, labels, metric="cosine"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_SIGS, "r", encoding="utf-8") as f:
        doc = json.load(f)

    class_labels: list[str] = doc["metadades"]["classes_usades"]
    voltes: list[dict] = doc["voltes"]
    n = len(voltes)
    d = len(class_labels)

    # Filtrar voltes buides (longitud_total == 0): no poden classificar-se
    valid_idx = [i for i, v in enumerate(voltes) if v["longitud_total"] > 0]
    skipped = n - len(valid_idx)

    # Matriu de signatures (només voltes vàlides)
    X = np.zeros((len(valid_idx), d), dtype=float)
    for row, i in enumerate(valid_idx):
        comp = voltes[i]["composicio_normalitzada"]
        X[row] = [comp[c] for c in class_labels]

    X_norm = l2_normalize(X)

    # Cerca de k òptim per silhouette
    k_results = []
    best = (-1.0, None, None)  # (score, k, labels)
    k_max_eff = min(K_MAX, len(valid_idx) - 1)
    for k in range(K_MIN, k_max_eff + 1):
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_norm)
        score = safe_silhouette(X_norm, labels)
        k_results.append({"k": k, "silhouette": round(score, 4)})
        if score > best[0]:
            best = (score, k, labels)

    best_score, best_k, labels = best
    km_final = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = km_final.fit_predict(X_norm)
    centroids = km_final.cluster_centers_  # en espai L2-normalitzat

    # Distàncies cosinus al centroide del propi clúster
    dist_to_own = np.zeros(len(valid_idx))
    for i in range(len(valid_idx)):
        c = centroids[labels[i]]
        # X_norm i centroides ja tenen norma ~1, però normalitzem per seguretat
        nu = np.linalg.norm(X_norm[i])
        nv = np.linalg.norm(c)
        cos = float(np.dot(X_norm[i], c) / (nu * nv + EPS))
        cos = max(-1.0, min(1.0, cos))
        dist_to_own[i] = 1.0 - cos

    # Z-score per detectar outliers dins de cada clúster
    z_scores = np.zeros(len(valid_idx))
    es_outlier = np.zeros(len(valid_idx), dtype=bool)
    for k in range(best_k):
        mask = labels == k
        if mask.sum() <= 1:
            continue
        mu = dist_to_own[mask].mean()
        sd = dist_to_own[mask].std()
        if sd < EPS:
            continue
        zs = (dist_to_own[mask] - mu) / sd
        z_scores[mask] = zs
        es_outlier[mask] = zs > 2.0

    # Mapatge volta_global_index -> info
    type_labels = [f"V{k}" for k in range(best_k)]
    assignments: dict[int, dict] = {}
    for row, i in enumerate(valid_idx):
        assignments[i] = {
            "tipus": type_labels[labels[row]],
            "dist_centroide": round(float(dist_to_own[row]), 6),
            "z_intra_tipus": round(float(z_scores[row]), 6),
            "es_outlier": bool(es_outlier[row]),
        }

    # Perfil de cada tipus VTx
    profile: dict[str, dict] = {}
    for k in range(best_k):
        mask = labels == k
        if mask.sum() == 0:
            continue
        # composicio mitjana en espai brut (no normalitzat L2) per llegibilitat
        comp_mitjana = X[mask].mean(axis=0)
        suma = comp_mitjana.sum()
        if suma > EPS:
            comp_mitjana = comp_mitjana / suma  # renormalitzar a fraccions
        # voltes pertanyents
        voltes_k = [valid_idx[j] for j in np.where(mask)[0]]
        teixits_k = Counter(voltes[i]["teixit"] for i in voltes_k)
        long_mitjana = float(np.mean([voltes[i]["longitud_total"] for i in voltes_k]))
        macros_mitjans = float(np.mean([voltes[i]["num_macros"] for i in voltes_k]))
        prof_mitjana = float(np.mean([voltes[i]["prof_max"] for i in voltes_k]))

        # Descriptors verticals (Tanda 3): mitjanes post-hoc, no entren al clustering
        nbv_vals  = [voltes[i].get("num_blocks_verticals", 0)   for i in voltes_k]
        snv_vals  = [voltes[i].get("sum_n_voltes_verticals", 0) for i in voltes_k]
        mnv_vals  = [voltes[i].get("max_n_voltes_verticals", 0) for i in voltes_k]
        fv_vals   = [voltes[i].get("frac_verticals", 0.0)       for i in voltes_k]
        frac_voltes_amb_vert = float(np.mean([1.0 if x > 0 else 0.0 for x in nbv_vals]))

        # Top-3 classes Cx dominants
        top_classes = sorted(
            zip(class_labels, comp_mitjana),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        profile[type_labels[k]] = {
            "num_voltes": int(mask.sum()),
            "composicio_classes_mitjana": {
                c: round(float(v), 4) for c, v in zip(class_labels, comp_mitjana)
            },
            "top_classes": [{"classe": c, "frac": round(float(v), 4)}
                            for c, v in top_classes],
            "long_mitjana": round(long_mitjana, 2),
            "num_macros_mitja": round(macros_mitjans, 2),
            "prof_max_mitjana": round(prof_mitjana, 2),
            # Descriptors verticals
            "frac_voltes_amb_verticals":    round(frac_voltes_amb_vert, 4),
            "num_blocks_verticals_mitja":   round(float(np.mean(nbv_vals)), 4),
            "sum_n_voltes_verticals_mitja": round(float(np.mean(snv_vals)), 4),
            "max_n_voltes_verticals_max":   int(max(mnv_vals) if mnv_vals else 0),
            "frac_verticals_mitja":         round(float(np.mean(fv_vals)), 4),
            "teixits_presents": dict(teixits_k.most_common()),
            "centroide_L2": [round(float(x), 6) for x in centroids[k]],
        }

    # Seqüències per teixit
    seq_per_teixit: dict[str, list[str]] = defaultdict(list)
    info_per_teixit: dict[str, list[dict]] = defaultdict(list)
    for i, v in enumerate(voltes):
        if i in assignments:
            tipus = assignments[i]["tipus"]
        else:
            tipus = "VOID"  # volta buida
        seq_per_teixit[v["teixit"]].append(tipus)
        info_per_teixit[v["teixit"]].append({
            "volta": v["volta"],
            "tipus": tipus,
            "longitud": v["longitud_total"],
        })

    # ---------- Escriure sortides ----------
    out_doc = {
        "metadades": {
            "num_voltes_total": n,
            "num_voltes_validades": len(valid_idx),
            "num_voltes_descartades_buides": skipped,
            "dim_signatura": d,
            "classes_token_usades": class_labels,
            "k_provats": k_results,
            "k_optim": best_k,
            "silhouette_optim": round(best_score, 4),
            "outliers_detectats": int(es_outlier.sum()),
        },
        "tipus_voltes": profile,
        "assignacions": [
            {
                "indice_global": i,
                "teixit": voltes[i]["teixit"],
                "volta": voltes[i]["volta"],
                **assignments[i],
            }
            for i in valid_idx
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teixit", "volta", "tipus", "dist_centroide",
                    "z_intra_tipus", "es_outlier", "longitud_total"])
        for i, v in enumerate(voltes):
            if i in assignments:
                a = assignments[i]
                w.writerow([v["teixit"], v["volta"], a["tipus"],
                            a["dist_centroide"], a["z_intra_tipus"],
                            int(a["es_outlier"]), v["longitud_total"]])
            else:
                w.writerow([v["teixit"], v["volta"], "VOID",
                            "", "", "", v["longitud_total"]])

    with open(OUT_SEQ, "w", encoding="utf-8") as f:
        json.dump({
            "metadades": {
                "alfabet": type_labels + (["VOID"] if skipped > 0 else []),
                "num_teixits": len(seq_per_teixit),
            },
            "sequencies": {
                t: {
                    "cadena": seq_per_teixit[t],
                    "longitud": len(seq_per_teixit[t]),
                    "voltes": info_per_teixit[t],
                }
                for t in seq_per_teixit
            },
        }, f, ensure_ascii=False, indent=2)

    # ---------- Resum a consola ----------
    print("FASE 3.4 COMPLETADA")
    print(f"  Voltes totals        : {n}")
    print(f"  Voltes validades     : {len(valid_idx)} (descartades buides: {skipped})")
    print(f"  k òptim (silhouette) : {best_k}  (score={round(best_score, 4)})")
    print(f"  Outliers (z>2)       : {int(es_outlier.sum())}")
    print("\nResum per tipus de volta:")
    for tipus, info in profile.items():
        top = " ".join(f"{x['classe']}={x['frac']:.2f}" for x in info["top_classes"])
        teixits_top = ", ".join(
            f"{t}({n})" for t, n in list(info["teixits_presents"].items())[:3]
        )
        print(f"  {tipus:<4} n={info['num_voltes']:<4} "
              f"long={info['long_mitjana']:>5.1f} "
              f"prof={info['prof_max_mitjana']:.2f}  "
              f"[{top}]  teixits: {teixits_top}")

    print("\nMostres de seqüències (primeres 20 voltes de cada teixit):")
    for t, seq in seq_per_teixit.items():
        mostra = " ".join(seq[:20])
        suf = " ..." if len(seq) > 20 else ""
        print(f"  {t:<22} [{len(seq):>3}]  {mostra}{suf}")

    print(f"\n  JSON tipus  : {OUT_JSON.relative_to(BASE.parent)}")
    print(f"  CSV         : {OUT_CSV.relative_to(BASE.parent)}")
    print(f"  Seqüències  : {OUT_SEQ.relative_to(BASE.parent)}")


if __name__ == "__main__":
    main()
