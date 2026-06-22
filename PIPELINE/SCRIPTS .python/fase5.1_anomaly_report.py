# -*- coding: utf-8 -*-
"""
FASE 5 — Report d'anomalies multi-canal
========================================

Agrega evidència de diversos canals d'anàlisi distribucional per identificar
voltes (i teixits) anòmals. Cada canal és un senyal independent; una volta
és més fiablement anòmala com més canals la marquen.

Canals d'entrada:
  1. Signatura distribucional (Fase 3.3):
        - z_intra_teixit          (distància al centroide del teixit)
        - z_salt_global           (salt cosinus respecte a la volta anterior)
        - frac_outlier            (fracció de tokens outlier dins la volta)
  2. Gramàtica MACRO (Fase 4.1):
        - surprisal_bits de la transició cap a aquesta volta
        - is_zero_prob_raw        (transició mai vista al corpus)
  3. Gramàtica MICRO (Fase 4.2):
        - mean_surprisal_bits     (H̄ de tokens dins la volta)
        - max_surprisal_bits      (token més inesperat)
        - num_zero_prob_raw_transitions
  4. Validació estadística de transicions (Fase 3.1b):
        - frac_trans_no_validades_FDR
          % de bigrames A→B adjacents a la volta que es van testar a
          Fase 3.1b i NO van passar el llindar FDR. Voltes que enllacen
          tokens via combinacions estadísticament febles són estructural-
          ment sospitoses.

Sortida:
  res_fase5/anomaly_report.json
  res_fase5/anomaly_report.csv   (rànquing de voltes)
  res_fase5/anomaly_teixits.csv  (agregat per teixit)
"""

from __future__ import annotations
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA

from config import (
    RES_FASE_5_1, ensure_dir,
    F2_INSTRUCTION_DATASET, F3_1B_TRANSITION_SIGNIF, F3_3_VOLTA_SIGNATURES,
    F4_1_MACRO_SURPRISAL_PER_VOLTA, F4_2_MICRO_GRAMMAR,
    F5_1_ANOMALY_REPORT_JSON, F5_1_ANOMALY_REPORT_CSV, F5_1_ANOMALY_TEIXITS_CSV,
    BASE,
)

# Entrades
IN_SIGS  = F3_3_VOLTA_SIGNATURES
IN_MICRO = F4_2_MICRO_GRAMMAR
IN_MACRO_CSV = F4_1_MACRO_SURPRISAL_PER_VOLTA
IN_SIGNIF    = F3_1B_TRANSITION_SIGNIF
IN_DATASET   = F2_INSTRUCTION_DATASET

# Sortides
OUT_DIR = RES_FASE_5_1
OUT_JSON = F5_1_ANOMALY_REPORT_JSON
OUT_CSV  = F5_1_ANOMALY_REPORT_CSV
OUT_CSV_TX = F5_1_ANOMALY_TEIXITS_CSV

# ─── Paràmetres del mètode ──────────────────────────────────────────
# El score d'anomalia s'obté com a primer component principal (PC1)
# d'una PCA sobre les senyals continues z-standarditzades dels 3 canals.
# Així s'elimina el pes arbitrari `1.5` que tenia la versió anterior:
# els pesos dels canals surten de la pròpia variància de les dades.
#
# Senyals que entren al PCA:
#   · z_intra_teixit            (signatura — Fase 3.3)
#   · z_salt_global             (signatura — Fase 3.3)
#   · z_frac_outlier            (signatura — Fase 3.3)
#   · z_surprisal_macro         (gramàtica macro — Fase 4.1)
#   · z_mean_surprisal_micro    (gramàtica micro — Fase 4.2)
#   · z_max_surprisal_micro     (gramàtica micro — Fase 4.2)
#   · z_zero_prob_macro         (indicador → standarditzat)
#   · z_num_zero_prob_micro     (count → standarditzat)
# La categoria qualitativa (CRÍTICA/ALTA/MODERADA/normal) es manté
# basada en el nombre de banderes binàries actives (z > Z_THRESHOLD).
Z_THRESHOLD = 2.0
RANDOM_STATE = 42
EPS = 1e-12


# ---------------------------------------------------------------------------
# Utilitats
# ---------------------------------------------------------------------------
def zscore_list(xs: list[float]) -> tuple[list[float], float, float]:
    if not xs:
        return [], 0.0, 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var)
    if sd < EPS:
        return [0.0] * len(xs), mu, sd
    return [(x - mu) / sd for x in xs], mu, sd


# ---------------------------------------------------------------------------
# Càrrega d'entrades
# ---------------------------------------------------------------------------
def load_signatures() -> dict[tuple[str, int], dict]:
    with open(IN_SIGS, "r", encoding="utf-8") as f:
        doc = json.load(f)
    out = {}
    for v in doc["voltes"]:
        out[(v["teixit"], v["volta"])] = {
            "z_intra_teixit": v.get("z_intra_teixit"),
            "z_salt_global": v.get("z_salt_global"),
            "frac_outlier": v.get("frac_outlier", 0.0),
            "dist_centroide_teixit": v.get("dist_centroide_teixit"),
        }
    return out


def load_macro() -> dict[tuple[str, int], dict]:
    """Llegeix el CSV de surprisal MACRO (una fila per posició de seqüència).
    Els EOS no corresponen a cap volta i es descarten."""
    out = {}
    with open(IN_MACRO_CSV, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            volta_raw = row.get("volta", "")
            if not volta_raw or volta_raw == "":
                continue
            try:
                volta = int(volta_raw)
            except ValueError:
                continue
            key = (row["teixit"], volta)
            surp = row["surprisal_bits"]
            out[key] = {
                "tipus_volta": row["tipus"],
                "context_macro": row["context"],
                "p_macro": float(row["p"]) if row["p"] else None,
                "surprisal_macro": float(surp) if surp else None,
                "context_seen_macro": bool(int(row["context_seen"])),
                "is_zero_prob_macro": bool(int(row["is_zero_prob_raw"])),
            }
    return out


def load_micro() -> dict[tuple[str, int], dict]:
    with open(IN_MICRO, "r", encoding="utf-8") as f:
        doc = json.load(f)
    out = {}
    for r in doc["per_volta"]:
        out[(r["teixit"], r["volta"])] = {
            "longitud_micro": r["longitud"],
            "mean_surprisal_micro": r.get("mean_surprisal_bits"),
            "max_surprisal_micro": r.get("max_surprisal_bits"),
            "perplexity_micro": r.get("perplexity"),
            "num_zero_prob_raw_micro": r.get("num_zero_prob_raw_transitions", 0),
        }
    return out


def load_transition_validity() -> dict[tuple[str, int], dict]:
    """
    Per cada volta, % de transicions adjacents A→B que han estat testades
    a Fase 3.1b i NO han passat el llindar FDR (q ≥ 0.05).

    - Només es comptabilitzen bigrames "testats" (els que tenen count≥3
      al corpus i, per tant, apareixen al fitxer de significància).
    - Bigrames rars no testats no contribueixen ni al numerador ni al
      denominador; però sí es comptabilitzen a `n_trans_no_testades` per
      traçabilitat.
    - Si una volta no té cap transició testada, els camps són None i no
      contribueixen al PCA.
    """
    with open(IN_SIGNIF, "r", encoding="utf-8") as f:
        signif = json.load(f)
    with open(IN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    validitat: dict[tuple[str, str], bool] = {
        (r["A"], r["B"]): bool(r["significatiu_FDR"])
        for r in signif["transicions"]
    }

    out: dict[tuple[str, int], dict] = {}
    for v in dataset:
        toks = v.get("tokens_expandits") or []
        n_test = n_inv = n_no_test = 0
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i + 1])
            if pair in validitat:
                n_test += 1
                if not validitat[pair]:
                    n_inv += 1
            else:
                n_no_test += 1
        frac = (n_inv / n_test) if n_test > 0 else None
        out[(v["teixit"], v["volta"])] = {
            "n_trans_testades":      n_test,
            "n_trans_no_validades":  n_inv,
            "n_trans_no_testades":   n_no_test,
            "frac_trans_no_validades": frac,
        }
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigs = load_signatures()
    macro = load_macro()
    micro = load_micro()
    validitat = load_transition_validity()

    keys = sorted(sigs.keys())

    # Recollir valors per Z-score global de cada canal continu
    surp_macro_vals = [macro.get(k, {}).get("surprisal_macro") for k in keys]
    surp_micro_vals = [micro.get(k, {}).get("mean_surprisal_micro") for k in keys]
    max_micro_vals  = [micro.get(k, {}).get("max_surprisal_micro") for k in keys]

    def _z(vals: list):
        valid_idx = [i for i, v in enumerate(vals) if v is not None]
        valid_vals = [vals[i] for i in valid_idx]
        zs, mu, sd = zscore_list(valid_vals)
        out = [None] * len(vals)
        for i, z in zip(valid_idx, zs):
            out[i] = z
        return out, mu, sd

    z_macro, mu_M, sd_M = _z(surp_macro_vals)
    z_mean_micro, mu_mi, sd_mi = _z(surp_micro_vals)
    z_max_micro, mu_mx, sd_mx = _z(max_micro_vals)

    # Senyals continues addicionals (Z-score sobre la població de voltes)
    frac_outlier_vals = [sigs[k].get("frac_outlier") for k in keys]
    z_frac_outlier, mu_fo, sd_fo = _z(frac_outlier_vals)

    zero_macro_vals = [
        1.0 if macro.get(k, {}).get("is_zero_prob_macro") else 0.0
        for k in keys
    ]
    z_zero_macro, _, _ = _z(zero_macro_vals)

    zero_micro_counts = [
        float(micro.get(k, {}).get("num_zero_prob_raw_micro", 0) or 0)
        for k in keys
    ]
    z_zero_micro, _, _ = _z(zero_micro_counts)

    # Fracció de transicions no validades FDR (Fase 3.1b)
    frac_inv_vals = [validitat.get(k, {}).get("frac_trans_no_validades") for k in keys]
    z_frac_inv, mu_inv, sd_inv = _z(frac_inv_vals)

    # Z d'intra-teixit i de salt: vénen ja com a z des de Fase 3.3
    z_intra_vals = [sigs[k].get("z_intra_teixit") for k in keys]
    z_salt_vals  = [sigs[k].get("z_salt_global") for k in keys]

    # Construcció de la matriu PCA (n × k) — substituïm None per 0
    def _zero_if_none(v):
        return 0.0 if v is None else float(v)

    canals = [
        ("z_intra_teixit",           z_intra_vals),
        ("z_salt_global",            z_salt_vals),
        ("z_frac_outlier",           z_frac_outlier),
        ("z_surprisal_macro",        z_macro),
        ("z_mean_surprisal_micro",   z_mean_micro),
        ("z_max_surprisal_micro",    z_max_micro),
        ("z_zero_prob_macro",        z_zero_macro),
        ("z_num_zero_prob_micro",    z_zero_micro),
        ("z_frac_trans_no_validades", z_frac_inv),
    ]
    M = np.array([
        [_zero_if_none(v) for _, vals in canals for v in [vals[i]]]
        for i in range(len(keys))
    ], dtype=np.float64)

    # Estandardització columna a columna (per si alguna columna tenia escala diferent)
    col_mean = M.mean(axis=0)
    col_std  = M.std(axis=0)
    col_std[col_std < EPS] = 1.0
    M_std = (M - col_mean) / col_std

    pca = PCA(n_components=min(M_std.shape[1], M_std.shape[0]), random_state=RANDOM_STATE)
    pca.fit(M_std)
    pc1_loadings = pca.components_[0]

    # Convenció de signe: PC1 ha de créixer amb senyals més anòmales.
    # Imposem que la suma de loadings sigui positiva (la majoria de
    # canals representen "més valor = més anòmal").
    if pc1_loadings.sum() < 0:
        pc1_loadings = -pc1_loadings

    pc1_scores = M_std @ pc1_loadings
    # Trasllat perquè el score mínim sigui 0 (PC1 té mitjana 0 per construcció).
    # Així Σscore per teixit és sempre ≥ 0 i interpretable com a "anomalia acumulada".
    pc1_scores = pc1_scores - pc1_scores.min()
    var_explicada_pc1 = float(pca.explained_variance_ratio_[0])

    # Construir registres per volta
    records: list[dict] = []
    for idx, key in enumerate(keys):
        teixit, volta = key
        s = sigs[key]
        m = macro.get(key, {})
        u = micro.get(key, {})
        v = validitat.get(key, {})

        signals = {
            # Canal 1: signatura
            "z_intra_teixit":  s.get("z_intra_teixit"),
            "z_salt_global":   s.get("z_salt_global"),
            "frac_outlier":    s.get("frac_outlier"),
            # Canal 2: macro
            "surprisal_macro": m.get("surprisal_macro"),
            "z_surprisal_macro": z_macro[idx],
            "is_zero_prob_macro": m.get("is_zero_prob_macro", False),
            # Canal 3: micro
            "mean_surprisal_micro": u.get("mean_surprisal_micro"),
            "max_surprisal_micro":  u.get("max_surprisal_micro"),
            "z_mean_surprisal_micro": z_mean_micro[idx],
            "z_max_surprisal_micro":  z_max_micro[idx],
            "num_zero_prob_raw_micro": u.get("num_zero_prob_raw_micro", 0),
            # Canal 4: validació estadística de transicions
            "n_trans_testades":      v.get("n_trans_testades", 0),
            "n_trans_no_validades":  v.get("n_trans_no_validades", 0),
            "frac_trans_no_validades": v.get("frac_trans_no_validades"),
            "z_frac_trans_no_validades": z_frac_inv[idx],
        }

        # Banderes binàries (canals "actius")
        flags = {
            "flag_sig_intra":   _ge(signals["z_intra_teixit"], Z_THRESHOLD),
            "flag_sig_salt":    _ge(signals["z_salt_global"], Z_THRESHOLD),
            "flag_macro_surp":  _ge(signals["z_surprisal_macro"], Z_THRESHOLD),
            "flag_macro_zero":  bool(signals["is_zero_prob_macro"]),
            "flag_micro_mean":  _ge(signals["z_mean_surprisal_micro"], Z_THRESHOLD),
            "flag_micro_max":   _ge(signals["z_max_surprisal_micro"], Z_THRESHOLD),
            "flag_micro_zero":  (signals["num_zero_prob_raw_micro"] or 0) > 0,
            "flag_trans_invalides": _ge(signals["z_frac_trans_no_validades"], Z_THRESHOLD),
        }
        num_flags = sum(1 for v in flags.values() if v)

        # Score combinat: primer component principal d'una PCA sobre
        # els canals continus z-standarditzats (els pesos surten de la
        # variància de les dades, no d'una fórmula arbitrària).
        score = float(pc1_scores[idx])

        # Categoria cualitativa
        if num_flags >= 3:
            category = "CRÍTICA"
        elif num_flags == 2:
            category = "ALTA"
        elif num_flags == 1:
            category = "MODERADA"
        else:
            category = "normal"

        # Si la transició és cross-canal (macro i micro tots dos actius)
        cross = (flags["flag_macro_surp"] or flags["flag_macro_zero"]) and \
                (flags["flag_micro_mean"] or flags["flag_micro_max"] or flags["flag_micro_zero"])

        records.append({
            "teixit": teixit,
            "volta":  volta,
            "tipus_volta": m.get("tipus_volta"),
            "context_macro": m.get("context_macro"),
            "longitud_micro": u.get("longitud_micro"),
            "score_anomalia": round(score, 4),
            "num_flags_actius": num_flags,
            "categoria": category,
            "cross_canal": cross,
            **{k: _round(v) for k, v in signals.items()},
            **flags,
        })

    # Ordenar pel score
    records.sort(key=lambda r: r["score_anomalia"], reverse=True)

    # Agregar per teixit
    per_teixit: dict[str, dict] = defaultdict(lambda: {
        "n_voltes": 0,
        "n_CRITICA": 0, "n_ALTA": 0, "n_MODERADA": 0,
        "score_total": 0.0, "score_max": 0.0,
        "n_cross_canal": 0,
    })
    for r in records:
        t = r["teixit"]
        per_teixit[t]["n_voltes"] += 1
        per_teixit[t]["score_total"] += r["score_anomalia"]
        per_teixit[t]["score_max"] = max(per_teixit[t]["score_max"], r["score_anomalia"])
        if r["categoria"] in ("CRÍTICA", "ALTA", "MODERADA"):
            per_teixit[t][f"n_{_strip(r['categoria'])}"] += 1
        if r["cross_canal"]:
            per_teixit[t]["n_cross_canal"] += 1
    for t, d in per_teixit.items():
        d["score_mitja"] = round(d["score_total"] / d["n_voltes"], 4)
        d["score_total"] = round(d["score_total"], 4)
        d["score_max"]  = round(d["score_max"], 4)

    # ------------------ Sortides ------------------
    canal_noms = [n for n, _ in canals]
    out_doc = {
        "parametres": {
            "Z_THRESHOLD": Z_THRESHOLD,
            "RANDOM_STATE": RANDOM_STATE,
            "metode_score": "PCA · PC1 sobre canals z-standarditzats",
            "canals_pca": canal_noms,
        },
        "metadades": {
            "num_voltes_avaluades": len(records),
            "canals_continus_grammar": {
                "surprisal_macro":      {"mu": round(mu_M, 4),  "sd": round(sd_M, 4)},
                "mean_surprisal_micro": {"mu": round(mu_mi, 4), "sd": round(sd_mi, 4)},
                "max_surprisal_micro":  {"mu": round(mu_mx, 4), "sd": round(sd_mx, 4)},
                "frac_trans_no_validades": {"mu": round(mu_inv, 4), "sd": round(sd_inv, 4)},
            },
            "pca": {
                "variancia_pc1": round(var_explicada_pc1, 4),
                "loadings_pc1": {
                    canal_noms[i]: round(float(pc1_loadings[i]), 4)
                    for i in range(len(canal_noms))
                },
                "tots_components_variancia": [
                    round(float(v), 4) for v in pca.explained_variance_ratio_
                ],
            },
            "categoria_regla": "CRÍTICA≥3 flags · ALTA=2 · MODERADA=1 · normal=0",
        },
        "anomalies": records,
        "agregat_per_teixit": per_teixit,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    # CSV principal
    cols = list(records[0].keys()) if records else []
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)

    # CSV per teixit
    with open(OUT_CSV_TX, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["teixit", "n_voltes", "n_CRITICA", "n_ALTA", "n_MODERADA",
                    "n_cross_canal", "score_total", "score_max", "score_mitja"])
        ordered = sorted(per_teixit.items(),
                         key=lambda kv: kv[1]["score_total"], reverse=True)
        for t, d in ordered:
            w.writerow([t, d["n_voltes"], d["n_CRITICA"], d["n_ALTA"],
                        d["n_MODERADA"], d["n_cross_canal"],
                        d["score_total"], d["score_max"], d["score_mitja"]])

    # ------------------ Resum ------------------
    print("FASE 5 COMPLETADA")
    print(f"  Voltes avaluades       : {len(records)}")
    by_cat = defaultdict(int)
    for r in records:
        by_cat[r["categoria"]] += 1
    for cat in ("CRÍTICA", "ALTA", "MODERADA", "normal"):
        print(f"    {cat:<10} : {by_cat[cat]}")
    n_cross = sum(1 for r in records if r["cross_canal"])
    print(f"  Anomalies cross-canal  : {n_cross}")

    print("\n  Top-10 voltes per score d'anomalia:")
    print(f"    {'teixit':<22} {'volta':>5}  {'tip':<4}  {'cat':<9}  {'score':>6}  flags")
    for r in records[:10]:
        actius = [k for k, v in r.items() if k.startswith("flag_") and v]
        flag_str = ",".join(a.replace("flag_", "") for a in actius)
        print(f"    {r['teixit']:<22} v{r['volta']:<4}  {r['tipus_volta'] or '-':<4}  "
              f"{r['categoria']:<9}  {r['score_anomalia']:>6.2f}  {flag_str}")

    print("\n  Ranking per teixit (score total):")
    ordered = sorted(per_teixit.items(),
                     key=lambda kv: kv[1]["score_total"], reverse=True)
    for t, d in ordered:
        print(f"    {t:<22}  Σscore={d['score_total']:>7.2f}  "
              f"max={d['score_max']:>5.2f}  C/A/M={d['n_CRITICA']}/{d['n_ALTA']}/{d['n_MODERADA']}  "
              f"cross={d['n_cross_canal']}")

    print(f"\n  JSON : {OUT_JSON.relative_to(BASE.parent)}")
    print(f"  CSV1 : {OUT_CSV.relative_to(BASE.parent)}")
    print(f"  CSV2 : {OUT_CSV_TX.relative_to(BASE.parent)}")


# Helpers
def _ge(x, t):
    return (x is not None) and (x >= t)

def _round(x):
    if isinstance(x, float):
        return round(x, 6)
    return x

def _strip(cat: str) -> str:
    return cat.replace("Í", "I").replace("í", "i")


if __name__ == "__main__":
    main()
