# -*- coding: utf-8 -*-
"""
FASE 5.1B — Validació criptogràfica i steganogràfica clàssica
=============================================================

Aplica diverses anàlisis criptoanalítiques sobre el corpus per **detectar
candidats a missatges amagats** (codis, marques d'autoria, anotacions
no instructives) dins el quadern de Carmen Machado.

Premissa metodològica
---------------------
L'orquestrador de Fase 5 prioritza voltes "anòmales" *des del punt de vista
de la gramàtica del teixit*: voltes de farciment llargues, transicions
inhabituals, surprisal alt. **Un missatge codificat amb intenció seria
dissenyat exactament perquè NO triggerés aquestes anomalies** (es camuflaria
imitant la sintaxi del corpus). Per tant, buscar codi *només* sobre les
voltes CRITICA/ALTA és anar a buscar les claus on hi ha llum.

Aquest mòdul fa quatre anàlisis complementàries:

  A. Test clàssic sobre voltes CRITICA+ALTA (control negatiu).
       Esperem que doni "no codi", confirmant que les anomalies de grammar
       són estructurals (farciment) i no lingüístiques (xifrat).

  B. Test clàssic sobre sub-stream de MARCADORS.
       Eliminem els tokens estructurals base (l, p, g, a, r, x) i analitzem
       només els marcadors: números, paréntesis, símbols, lletres aïllades.
       Aquest és el lloc natural on viuria un codi amagat.

  C. Concentració de tokens RARE per volta i per teixit.
       Els 52 tokens hapax (freq=1) són els candidats més obvis a marcadors
       únics (D, v, {, }, números soltos). Llistem les voltes que en
       contenen per inspecció manual posterior.

  D. Projecció binària + test de runs de Wald-Wolfowitz.
       Projectem la seqüència completa a {0 = punt base 'l', 1 = qualsevol
       altre}. Si la distribució de "no-l" és aleatòria, el nombre de runs
       segueix una normal coneguda. Desviacions fortes (|z|>3) indiquen
       agrupacions no aleatòries → possibles "paraules" codificades.

Entrades:
    res_fase5.1/anomaly_report.json       (categories per volta)
    res_fase2/instruction_dataset.json  (seqüències de tokens per volta)
    res_fase3.2/distributional_classes.json (etiqueta C_RARE)

Sortides:
    res_fase5.1b/crypto_validation.json
    res_fase5.1b/crypto_validation.csv
    res_fase5.1b/candidats_steganografia.csv   (rànquing de voltes sospitoses)

El veredicte final NO és un sí/no, sinó un MAPA DE CANDIDATS prioritzat
per la convergència de les anàlisis A–D. La inspecció qualitativa final
del facsímil correspon a la fase humana (no estadística).
"""

from __future__ import annotations
import json
import csv
import math
import random
from collections import Counter, defaultdict
from functools import reduce
from pathlib import Path

from config import (
    BASE, RES_FASE_5_1B, ensure_dir, RANDOM_STATE,
    F2_INSTRUCTION_DATASET, F3_2_DISTRIB_CLASSES,
    F5_1_ANOMALY_REPORT_JSON, F5_1B_CRYPTO_VALIDATION, F5_1B_CANDIDATS_STEG_CSV,
)

# Entrades
IN_ANOMALIES = F5_1_ANOMALY_REPORT_JSON
IN_DATASET   = F2_INSTRUCTION_DATASET
IN_CLASSES   = F3_2_DISTRIB_CLASSES

# Sortides
OUT_DIR      = RES_FASE_5_1B
OUT_JSON     = F5_1B_CRYPTO_VALIDATION
OUT_CSV      = OUT_DIR / "crypto_validation.csv"
OUT_CSV_CAND = F5_1B_CANDIDATS_STEG_CSV

# ─── Paràmetres ──────────────────────────────────────────────────────────
CATEGORIES_FILTRADES = ["CRÍTICA", "ALTA"]
# Tokens estructurals base (els 6 tipus de punt que dominen el corpus).
# Tota la resta es considera "marcador" per l'anàlisi B.
TOKENS_ESTRUCTURALS  = {"l", "p", "g", "a", "r", "x"}
KASISKI_NGRAM_SIZES  = [3, 4, 5]
KASISKI_MIN_REPEATS  = 2
KASISKI_GCD_MIN_DIST = 5
IC_TOLERANCE_REL     = 0.15
TOP_FREQ_N           = 20
RANDOM_STATE         = 42
N_PERMUTACIONS_NULL  = 1000
RUNS_Z_THRESHOLD     = 3.0          # |z|>3 = desviació molt forta
MIN_TOKENS_VOLTA_RUNS = 20          # voltes massa curtes no es testen
EPS                  = 1e-12


# ═════════════════════════════════════════════════════════════════════════
# Primitives criptoanalítiques
# ═════════════════════════════════════════════════════════════════════════

def index_of_coincidence(tokens: list[str]) -> float:
    """IC = Σ f_i (f_i − 1) / (N (N − 1))."""
    N = len(tokens)
    if N < 2:
        return 0.0
    counts = Counter(tokens)
    num = sum(c * (c - 1) for c in counts.values())
    den = N * (N - 1)
    return num / den


def frequency_analysis(tokens: list[str], top_n: int = TOP_FREQ_N) -> dict:
    N = len(tokens)
    counts = Counter(tokens)
    top = counts.most_common(top_n)
    return {
        "n_tokens": N,
        "alphabet_size": len(counts),
        "top": [
            {"token": tok, "freq": c, "pct": round(100.0 * c / max(N, 1), 3)}
            for tok, c in top
        ],
    }


def kasiski_test(tokens: list[str],
                 ngram_sizes: list[int] = KASISKI_NGRAM_SIZES,
                 min_repeats: int = KASISKI_MIN_REPEATS) -> dict:
    """N-grames repetits + GCD de distàncies entre repeticions."""
    resultats = {}
    for n in ngram_sizes:
        positions = defaultdict(list)
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            positions[gram].append(i)

        repeats = {g: pos for g, pos in positions.items()
                   if len(pos) >= min_repeats}

        distances = []
        for g, pos in repeats.items():
            for j in range(1, len(pos)):
                d = pos[j] - pos[j - 1]
                if d > 1:
                    distances.append(d)

        gcd_global = reduce(math.gcd, distances) if distances else 0
        factor_hist = Counter()
        for d in distances:
            for k in range(2, 11):
                if d % k == 0:
                    factor_hist[k] += 1

        top_repeat = None
        if repeats:
            top_g, top_pos = max(repeats.items(), key=lambda kv: len(kv[1]))
            top_repeat = {
                "ngram": list(top_g),
                "n_repeticions": len(top_pos),
                "primeres_posicions": top_pos[:5],
            }

        resultats[f"n{n}"] = {
            "n_grames_repetits": len(repeats),
            "n_distancies": len(distances),
            "gcd_global": gcd_global,
            "factor_hist_2a10": dict(sorted(factor_hist.items())),
            "top_repetit": top_repeat,
        }
    return resultats


def stream_summary(stream: list[str], etiqueta: str) -> dict:
    """Resum complet d'un stream: IC, IC_norm, IC_unif, freq, Kasiski."""
    N = len(stream)
    if N < 10:
        return {
            "etiqueta": etiqueta,
            "n_tokens": N,
            "nota": "Stream massa curt (<10 tokens) per IC fiable.",
        }
    freq = frequency_analysis(stream)
    V = freq["alphabet_size"]
    ic = index_of_coincidence(stream)
    ic_unif = 1.0 / V if V > 0 else 0.0
    ic_norm = ic * V
    kasiski = kasiski_test(stream)
    return {
        "etiqueta": etiqueta,
        "n_tokens": N,
        "alphabet_size": V,
        "IC": round(ic, 6),
        "IC_norm": round(ic_norm, 4),
        "IC_referencia_uniforme": round(ic_unif, 6),
        "IC_vs_uniforme_ratio": round(ic / ic_unif, 3) if ic_unif > 0 else None,
        "frequencies": freq,
        "kasiski": kasiski,
    }


def null_distribution_IC(stream: list[str],
                          n_perm: int = N_PERMUTACIONS_NULL,
                          seed: int = RANDOM_STATE) -> dict:
    """Sanity check: IC ha de ser invariant a permutació (sd ~ 0)."""
    rng = random.Random(seed)
    base = list(stream)
    obs = index_of_coincidence(base)
    perms = []
    for _ in range(n_perm):
        rng.shuffle(base)
        perms.append(index_of_coincidence(base))
    mean = sum(perms) / len(perms)
    var = sum((p - mean) ** 2 for p in perms) / len(perms)
    sd = math.sqrt(var)
    return {
        "n_perm": n_perm,
        "IC_observat": round(obs, 6),
        "IC_mean_perm": round(mean, 6),
        "IC_sd_perm": round(sd, 8),
        "nota": ("L'IC és invariant a permutacions; el SD ha de ser ~0. "
                 "Aquesta verificació confirma que el test mesura "
                 "freqüències marginals, no ordre."),
    }


def runs_test_wald_wolfowitz(binary_seq: list[int]) -> dict:
    """Test de runs sobre una seqüència binària.

    Sota H0 (aleatorietat), si n0 = #zeros, n1 = #uns, n = n0+n1, R = #runs:
        E[R]   = 2·n0·n1 / n + 1
        Var[R] = 2·n0·n1·(2·n0·n1 − n) / (n²·(n−1))
        z      = (R − E[R]) / sqrt(Var[R])
    """
    n = len(binary_seq)
    n1 = sum(binary_seq)
    n0 = n - n1
    if n0 == 0 or n1 == 0 or n < 4:
        return {
            "n": n, "n0": n0, "n1": n1,
            "nota": "Seqüència no testejable (massa curta o constant).",
        }
    R = 1
    for i in range(1, n):
        if binary_seq[i] != binary_seq[i - 1]:
            R += 1
    mu = (2 * n0 * n1) / n + 1
    var_num = 2 * n0 * n1 * (2 * n0 * n1 - n)
    var_den = (n ** 2) * (n - 1)
    var = var_num / var_den if var_den > 0 else 0.0
    sd = math.sqrt(var) if var > 0 else 0.0
    z = (R - mu) / sd if sd > 0 else 0.0
    return {
        "n": n, "n0": n0, "n1": n1,
        "n_runs_observat": R,
        "n_runs_esperat": round(mu, 4),
        "sd": round(sd, 4),
        "z": round(z, 4),
        "interpretacio": (
            "menys runs dels esperats (agrupacio)" if z < -RUNS_Z_THRESHOLD
            else "mes runs dels esperats (alternancia)" if z > RUNS_Z_THRESHOLD
            else "compatible amb aleatorietat"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════
# Veredictes per secció
# ═════════════════════════════════════════════════════════════════════════

def veredicte_A(anom: dict, corpus: dict) -> dict:
    """Secció A: test clàssic sobre voltes CRITICA+ALTA.
    Esperem rebuig — confirma que les anomalies de grammar són farciment.
    """
    if "IC" not in anom or "IC" not in corpus:
        return {"diagnostic": "dades insuficients"}
    ic_a, ic_c = anom["IC"], corpus["IC"]
    ratio = ic_a / ic_c if ic_c > 0 else float("inf")
    iguals = abs(ratio - 1.0) <= IC_TOLERANCE_REL
    periodicitat_anom = any(
        k["gcd_global"] > 1 and k["n_distancies"] >= KASISKI_GCD_MIN_DIST
        for k in anom["kasiski"].values()
    )

    if iguals:
        diagn = "mateix règim que corpus"
        nota = ("Cap signatura de xifrat. Esperat: les anomalies de grammar "
                "són estructurals, no lingüístiques.")
    elif ratio > 1.0 + IC_TOLERANCE_REL:
        diagn = "concentració extrema (anom MÉS zipfià que corpus)"
        nota = ("Direcció OPOSADA a la d'un xifrat (que aplanaria la distribució). "
                "Les voltes anòmales són farciment dominat per un únic token base. "
                "Aquest resultat confirma la hipòtesi metodològica: un codi amagat "
                "no es manifesta com a anomalia de grammar — es camufla.")
    else:
        diagn = "IC més baix que corpus"
        nota = (f"Compatibilitat dèbil amb monoalfabètic; Kasiski "
                f"{'detecta' if periodicitat_anom else 'no detecta'} periodicitat.")

    return {
        "diagnostic": diagn,
        "IC_anomalies": ic_a,
        "IC_corpus": ic_c,
        "ratio": round(ratio, 4),
        "periodicitat_kasiski": periodicitat_anom,
        "nota_interpretativa": nota,
    }


def veredicte_B(marc: dict) -> dict:
    """Secció B: sub-stream de marcadors. Aquí sí busquem evidència positiva."""
    if "IC" not in marc:
        return {"diagnostic": "stream insuficient", "nota": marc.get("nota")}
    ic = marc["IC"]
    ic_unif = marc["IC_referencia_uniforme"]
    ratio_unif = ic / ic_unif if ic_unif > 0 else float("inf")
    periodicitat = [
        (n_key, k["gcd_global"], k["n_distancies"])
        for n_key, k in marc["kasiski"].items()
        if k["gcd_global"] > 1 and k["n_distancies"] >= KASISKI_GCD_MIN_DIST
    ]
    top1 = marc["frequencies"]["top"][0] if marc["frequencies"]["top"] else None
    top1_pct = top1["pct"] if top1 else 0

    indicadors = []
    if periodicitat:
        indicadors.append(
            f"periodicitat Kasiski (gcd>1, ≥{KASISKI_GCD_MIN_DIST} distàncies) "
            f"a {[p[0] for p in periodicitat]}"
        )
    if top1_pct > 50:
        indicadors.append(
            f"token dominant '{top1['token']}' "
            f"al {top1_pct:.1f}% del stream de marcadors"
        )
    if 5 < ratio_unif < 20:
        indicadors.append(
            f"IC moderadament estructurat (ratio vs unif = {ratio_unif:.1f})"
        )

    diagn = ("evidència positiva" if indicadors
             else "compatible amb notació tècnica natural")
    return {
        "diagnostic": diagn,
        "IC_marcadors": ic,
        "IC_marcadors_norm": marc["IC_norm"],
        "IC_uniforme": ic_unif,
        "ratio_vs_uniforme": round(ratio_unif, 4),
        "indicadors_detectats": indicadors,
        "n_indicadors": len(indicadors),
    }


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ───────── Carrega dades ────────────────────────────────────────────
    with open(IN_ANOMALIES, "r", encoding="utf-8") as f:
        rep = json.load(f)
    anomalies = rep["anomalies"]

    with open(IN_DATASET, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    with open(IN_CLASSES, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    tokens_rare = sorted([
        tok for tok, info in classes_data["tokens"].items()
        if info.get("classe") == "C_RARE"
    ])
    set_rare = set(tokens_rare)

    # Index (teixit, volta) → tokens
    idx_tokens = {}
    for inst in instructions:
        key = (inst["teixit"], inst["volta"])
        toks = inst.get("tokens_expandits", []) or []
        idx_tokens.setdefault(key, []).extend(toks)

    # Stream corpus complet
    stream_corpus = []
    for inst in instructions:
        stream_corpus.extend(inst.get("tokens_expandits", []) or [])

    # Voltes anòmales CRITICA+ALTA
    voltes_anomales = [v for v in anomalies if v.get("categoria") in CATEGORIES_FILTRADES]
    voltes_anomales.sort(
        key=lambda v: (-v.get("score_anomalia", 0.0), v["teixit"], v["volta"])
    )

    # Stream voltes anòmales
    stream_anom = []
    per_teixit_anom = defaultdict(list)
    detall_voltes_anom = []
    for v in voltes_anomales:
        key = (v["teixit"], v["volta"])
        toks = idx_tokens.get(key, [])
        stream_anom.extend(toks)
        per_teixit_anom[v["teixit"]].extend(toks)
        detall_voltes_anom.append({
            "teixit": v["teixit"],
            "volta": v["volta"],
            "categoria": v["categoria"],
            "score_anomalia": round(v.get("score_anomalia", 0.0), 4),
            "num_flags_actius": v.get("num_flags_actius"),
            "n_tokens": len(toks),
            "tokens": toks,
        })

    # ─────────────────────────────────────────────────────────────────────
    # SECCIÓ A — Test clàssic sobre voltes CRITICA+ALTA (control negatiu)
    # ─────────────────────────────────────────────────────────────────────
    resum_corpus = stream_summary(stream_corpus, "corpus_complet")
    resum_anom   = stream_summary(stream_anom,   "anomalies_global")
    null_anom    = (null_distribution_IC(stream_anom)
                    if len(stream_anom) >= 10 else None)
    resum_anom_per_teixit = {
        tx: stream_summary(toks, f"anomalies_{tx}")
        for tx, toks in per_teixit_anom.items()
    }
    ver_A = veredicte_A(resum_anom, resum_corpus)

    # ─────────────────────────────────────────────────────────────────────
    # SECCIÓ B — Sub-stream de MARCADORS
    # ─────────────────────────────────────────────────────────────────────
    def filtra_marcadors(seq):
        return [t for t in seq if t not in TOKENS_ESTRUCTURALS]

    stream_corpus_marc = filtra_marcadors(stream_corpus)
    stream_anom_marc   = filtra_marcadors(stream_anom)
    resum_corpus_marc = stream_summary(stream_corpus_marc, "marcadors_corpus")
    resum_anom_marc   = stream_summary(stream_anom_marc,   "marcadors_anomalies")
    ver_B = veredicte_B(resum_corpus_marc)

    marcadors_per_teixit = defaultdict(list)
    for inst in instructions:
        toks_m = filtra_marcadors(inst.get("tokens_expandits", []) or [])
        marcadors_per_teixit[inst["teixit"]].extend(toks_m)
    resum_marc_per_teixit = {
        tx: stream_summary(toks, f"marcadors_{tx}")
        for tx, toks in marcadors_per_teixit.items()
    }

    # ─────────────────────────────────────────────────────────────────────
    # SECCIÓ C — Concentració de tokens RARE
    # ─────────────────────────────────────────────────────────────────────
    voltes_amb_rare = []
    for inst in instructions:
        toks = inst.get("tokens_expandits", []) or []
        rares_en_volta = [t for t in toks if t in set_rare]
        if rares_en_volta:
            voltes_amb_rare.append({
                "teixit": inst["teixit"],
                "volta": inst["volta"],
                "n_tokens_volta": len(toks),
                "n_rare": len(rares_en_volta),
                "tokens_rare": rares_en_volta,
            })
    voltes_amb_rare.sort(key=lambda x: (-x["n_rare"], -x["n_tokens_volta"]))
    rare_per_teixit = Counter()
    for v in voltes_amb_rare:
        rare_per_teixit[v["teixit"]] += v["n_rare"]

    # ─────────────────────────────────────────────────────────────────────
    # SECCIÓ D — Projecció binària + runs test
    # ─────────────────────────────────────────────────────────────────────
    bin_global = [0 if t == "l" else 1 for t in stream_corpus]
    runs_global = runs_test_wald_wolfowitz(bin_global)

    runs_per_volta = []
    n_testejables = 0
    for inst in instructions:
        toks = inst.get("tokens_expandits", []) or []
        if len(toks) < MIN_TOKENS_VOLTA_RUNS:
            continue
        n_testejables += 1
        bseq = [0 if t == "l" else 1 for t in toks]
        rt = runs_test_wald_wolfowitz(bseq)
        if "z" in rt and abs(rt["z"]) > RUNS_Z_THRESHOLD:
            runs_per_volta.append({
                "teixit": inst["teixit"],
                "volta": inst["volta"],
                "n_tokens": len(toks),
                "z_runs": rt["z"],
                "n_runs_obs": rt["n_runs_observat"],
                "n_runs_esp": rt["n_runs_esperat"],
                "interpretacio": rt["interpretacio"],
                "pct_no_base": round(100.0 * rt["n1"] / rt["n"], 2),
            })
    runs_per_volta.sort(key=lambda x: -abs(x["z_runs"]))

    # ─────────────────────────────────────────────────────────────────────
    # CANDIDATS A STEGANOGRAFIA — convergència de senyals
    # ─────────────────────────────────────────────────────────────────────
    candidats = {}
    def init_cand(teixit, volta):
        k = (teixit, volta)
        if k not in candidats:
            candidats[k] = {
                "teixit": teixit, "volta": volta,
                "te_rare": False, "tokens_rare": [],
                "z_runs": None, "z_runs_abs": 0.0, "interpretacio_runs": None,
                "categoria_fase5": None, "score_anomalia": None,
            }
        return candidats[k]

    for v in voltes_amb_rare:
        c = init_cand(v["teixit"], v["volta"])
        c["te_rare"] = True
        c["tokens_rare"] = v["tokens_rare"]
    for r in runs_per_volta:
        c = init_cand(r["teixit"], r["volta"])
        c["z_runs"] = r["z_runs"]
        c["z_runs_abs"] = abs(r["z_runs"])
        c["interpretacio_runs"] = r["interpretacio"]
    for a in anomalies:
        if a.get("categoria") in CATEGORIES_FILTRADES:
            k = (a["teixit"], a["volta"])
            if k in candidats:
                c = candidats[k]
                c["categoria_fase5"] = a["categoria"]
                c["score_anomalia"] = round(a.get("score_anomalia", 0.0), 4)

    # Prioritat: cada RARE val 10 punts, |z_runs| s'hi suma
    candidats_llista = list(candidats.values())
    for c in candidats_llista:
        c["n_rare"] = len(c["tokens_rare"])
        c["prioritat"] = c["n_rare"] * 10 + c["z_runs_abs"]
    candidats_llista.sort(key=lambda x: -x["prioritat"])

    # Convergència triple: voltes que apareixen a A i C i D
    convergencia_triple = [
        c for c in candidats_llista
        if c["te_rare"] and c["z_runs"] is not None and c["categoria_fase5"]
    ]

    # ─────────────────────────────────────────────────────────────────────
    # Empaqueta JSON
    # ─────────────────────────────────────────────────────────────────────
    out = {
        "parametres": {
            "categories_filtrades": CATEGORIES_FILTRADES,
            "tokens_estructurals": sorted(TOKENS_ESTRUCTURALS),
            "kasiski_ngram_sizes": KASISKI_NGRAM_SIZES,
            "kasiski_min_repeats": KASISKI_MIN_REPEATS,
            "kasiski_gcd_min_dist": KASISKI_GCD_MIN_DIST,
            "IC_tolerance_rel": IC_TOLERANCE_REL,
            "runs_z_threshold": RUNS_Z_THRESHOLD,
            "min_tokens_volta_runs": MIN_TOKENS_VOLTA_RUNS,
            "top_freq_n": TOP_FREQ_N,
            "random_state": RANDOM_STATE,
            "n_permutacions_null": N_PERMUTACIONS_NULL,
        },
        "metadades": {
            "n_voltes_totals_avaluades_fase5": len(anomalies),
            "n_voltes_anomales_seleccionades": len(voltes_anomales),
            "n_tokens_stream_corpus": len(stream_corpus),
            "n_tokens_stream_anomalies": len(stream_anom),
            "n_tokens_marcadors_corpus": len(stream_corpus_marc),
            "n_tokens_marcadors_anomalies": len(stream_anom_marc),
            "pct_marcadors_corpus": round(
                100.0 * len(stream_corpus_marc) / max(len(stream_corpus), 1), 2
            ),
            "n_tokens_rare_vocabulari": len(set_rare),
            "n_voltes_amb_token_rare": len(voltes_amb_rare),
            "n_voltes_testejables_runs": n_testejables,
            "n_voltes_amb_runs_anomalous": len(runs_per_volta),
            "n_candidats_total": len(candidats_llista),
            "n_convergencia_triple_ACD": len(convergencia_triple),
        },
        "seccio_A_classic_sobre_anomalies_fase5": {
            "corpus_complet": resum_corpus,
            "anomalies_global": resum_anom,
            "anomalies_per_teixit": resum_anom_per_teixit,
            "control_null_IC_anomalies": null_anom,
            "veredicte": ver_A,
        },
        "seccio_B_sub_stream_marcadors": {
            "marcadors_corpus": resum_corpus_marc,
            "marcadors_anomalies": resum_anom_marc,
            "marcadors_per_teixit": resum_marc_per_teixit,
            "veredicte": ver_B,
        },
        "seccio_C_concentracio_tokens_rare": {
            "vocabulari_rare": tokens_rare,
            "voltes_amb_rare": voltes_amb_rare,
            "agregat_per_teixit": dict(sorted(
                rare_per_teixit.items(), key=lambda kv: -kv[1]
            )),
        },
        "seccio_D_runs_test": {
            "runs_global_corpus": runs_global,
            "voltes_amb_runs_anomalous": runs_per_volta,
        },
        "candidats_steganografia": candidats_llista,
        "convergencia_triple_A_C_D": convergencia_triple,
        "voltes_anomales_detall": detall_voltes_anom,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # CSV resum streams (A+B)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "stream", "n_tokens", "alphabet_size", "IC", "IC_norm",
            "IC_uniforme", "IC_ratio_vs_unif",
            "top_token", "top_token_pct",
            "kasiski_n3_repeats", "kasiski_n3_gcd",
            "kasiski_n4_repeats", "kasiski_n4_gcd",
        ])

        def row_for(r):
            if r.get("nota"):
                return [r.get("etiqueta", ""), r["n_tokens"], "", "", "",
                        "", "", "", "", "", "", "", ""]
            top = r["frequencies"]["top"][0] if r["frequencies"]["top"] else {}
            k3 = r["kasiski"].get("n3", {})
            k4 = r["kasiski"].get("n4", {})
            return [
                r["etiqueta"], r["n_tokens"], r["alphabet_size"],
                r["IC"], r["IC_norm"],
                r["IC_referencia_uniforme"], r["IC_vs_uniforme_ratio"],
                top.get("token", ""), top.get("pct", ""),
                k3.get("n_grames_repetits", ""), k3.get("gcd_global", ""),
                k4.get("n_grames_repetits", ""), k4.get("gcd_global", ""),
            ]

        w.writerow(row_for(resum_corpus))
        w.writerow(row_for(resum_anom))
        w.writerow(row_for(resum_corpus_marc))
        w.writerow(row_for(resum_anom_marc))
        for tx in sorted(resum_anom_per_teixit):
            w.writerow(row_for(resum_anom_per_teixit[tx]))
        for tx in sorted(resum_marc_per_teixit):
            w.writerow(row_for(resum_marc_per_teixit[tx]))

    # CSV candidats steganografia (per inspecció manual)
    with open(OUT_CSV_CAND, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "prioritat", "teixit", "volta",
            "n_rare", "tokens_rare",
            "z_runs", "interpretacio_runs",
            "categoria_fase5", "score_anomalia",
        ])
        for c in candidats_llista:
            w.writerow([
                round(c["prioritat"], 3),
                c["teixit"], c["volta"],
                c["n_rare"],
                " | ".join(c["tokens_rare"]) if c["tokens_rare"] else "",
                c["z_runs"] if c["z_runs"] is not None else "",
                c["interpretacio_runs"] or "",
                c["categoria_fase5"] or "",
                c["score_anomalia"] if c["score_anomalia"] is not None else "",
            ])

    # ─────────────────────────────────────────────────────────────────────
    # Resum llegible per consola
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 78)
    print("FASE 5.1B — VALIDACIO CRIPTOGRAFICA I STEGANOGRAFICA CLASSICA")
    print("=" * 78)

    print(f"\nCorpus complet : {len(stream_corpus)} tokens, "
          f"{resum_corpus['alphabet_size']} types")
    print(f"Anomalies (CRITICA+ALTA): {len(voltes_anomales)} voltes, "
          f"{len(stream_anom)} tokens")
    print(f"Marcadors (excloent {sorted(TOKENS_ESTRUCTURALS)}): "
          f"{len(stream_corpus_marc)} tokens corpus "
          f"({out['metadades']['pct_marcadors_corpus']}%), "
          f"{len(stream_anom_marc)} tokens anomalies")
    print(f"Vocabulari RARE: {len(set_rare)} tokens hapax")

    print("\n" + "-" * 78)
    print("SECCIO A — Test classic sobre anomalies (control negatiu)")
    print("-" * 78)
    if "IC" in resum_anom:
        print(f"  IC corpus    : {resum_corpus['IC']:.4f}")
        print(f"  IC anomalies : {resum_anom['IC']:.4f}  "
              f"(ratio {ver_A['ratio']:.2f})")
    print(f"  Diagnostic   : {ver_A['diagnostic']}")
    print(f"  {ver_A.get('nota_interpretativa','')}")

    print("\n" + "-" * 78)
    print("SECCIO B — Sub-stream de MARCADORS (on hauria de viure el codi)")
    print("-" * 78)
    if "IC" in resum_corpus_marc:
        print(f"  IC marcadors corpus    : {resum_corpus_marc['IC']:.4f} "
              f"(V={resum_corpus_marc['alphabet_size']}, "
              f"IC_norm={resum_corpus_marc['IC_norm']:.2f})")
        print(f"  IC uniforme referencia : {resum_corpus_marc['IC_referencia_uniforme']:.4f}")
        print(f"  Top 5 marcadors        : "
              f"{[(t['token'], t['pct']) for t in resum_corpus_marc['frequencies']['top'][:5]]}")
        for n_key in ["n3", "n4", "n5"]:
            k = resum_corpus_marc["kasiski"][n_key]
            print(f"  Kasiski {n_key}: repeats={k['n_grames_repetits']:>3} "
                  f"gcd={k['gcd_global']:>3} n_dist={k['n_distancies']:>4} "
                  f"top_factors={dict(list(k['factor_hist_2a10'].items())[:4])}")
    print(f"  Diagnostic   : {ver_B['diagnostic']}")
    for ind in ver_B.get("indicadors_detectats", []):
        print(f"    - {ind}")

    print("\n" + "-" * 78)
    print("SECCIO C — Concentracio de tokens RARE")
    print("-" * 78)
    print(f"  Voltes amb >=1 RARE: {len(voltes_amb_rare)}")
    print(f"  Top 10 voltes per nombre de tokens RARE:")
    for v in voltes_amb_rare[:10]:
        print(f"    {v['teixit']:20s} v{v['volta']:>3}  "
              f"n_rare={v['n_rare']}  tokens={v['tokens_rare']}")
    print(f"  Distribucio per teixit:")
    for tx, n in sorted(rare_per_teixit.items(), key=lambda kv: -kv[1]):
        print(f"    {tx:25s}: {n}")

    print("\n" + "-" * 78)
    print("SECCIO D — Test de runs (Wald-Wolfowitz)")
    print("-" * 78)
    print(f"  Global corpus (l vs no-l):")
    print(f"    n={runs_global['n']} n0(l)={runs_global['n0']} "
          f"n1(no-l)={runs_global['n1']}")
    print(f"    runs_obs={runs_global['n_runs_observat']} "
          f"runs_esp={runs_global['n_runs_esperat']:.1f} "
          f"z={runs_global['z']:.2f}  -> {runs_global['interpretacio']}")
    print(f"  Voltes amb |z|>{RUNS_Z_THRESHOLD}: {len(runs_per_volta)} "
          f"(de {n_testejables} testejables)")
    for r in runs_per_volta[:10]:
        print(f"    {r['teixit']:20s} v{r['volta']:>3}  "
              f"n={r['n_tokens']:>4} z={r['z_runs']:>+6.2f}  "
              f"{r['interpretacio']}")

    print("\n" + "=" * 78)
    print("CANDIDATS A STEGANOGRAFIA — convergencia de senyals (top 20)")
    print("=" * 78)
    print(f"  Total candidats: {len(candidats_llista)}  |  "
          f"Convergencia triple A+C+D: {len(convergencia_triple)}")
    print(f"  {'teixit':22s} {'volta':>5s} {'rare':>5s} {'z_runs':>8s}  "
          f"{'cat_f5':>9s} tokens_rare")
    for c in candidats_llista[:20]:
        z_str = f"{c['z_runs']:+.2f}" if c['z_runs'] is not None else "  ·  "
        cat   = c['categoria_fase5'] or "—"
        toks  = " ".join(c['tokens_rare'][:3]) if c['tokens_rare'] else ""
        if len(c['tokens_rare']) > 3:
            toks += f" (+{len(c['tokens_rare'])-3})"
        print(f"  {c['teixit']:22s} {c['volta']:>5d} {c['n_rare']:>5d} {z_str:>8s}  "
              f"{cat:>9s} {toks}")

    if convergencia_triple:
        print("\n" + "*" * 78)
        print("CONVERGENCIA TRIPLE A+C+D (maxima prioritat per inspeccio)")
        print("*" * 78)
        for c in convergencia_triple:
            print(f"  {c['teixit']} v{c['volta']}  "
                  f"n_rare={c['n_rare']}  z={c['z_runs']:+.2f}  "
                  f"cat={c['categoria_fase5']}  score={c['score_anomalia']}")
            print(f"      tokens_rare: {c['tokens_rare']}")

    print("\n" + "-" * 78)
    print("CONCLUSIO METODOLOGICA")
    print("-" * 78)
    print(
        "  · Seccio A: les voltes prioritzades per fase5 NO contenen signatura\n"
        "    de xifrat (com era esperable: un codi amagat es dissenya per NO\n"
        "    triggerejar anomalies de grammar).\n"
        "  · Seccions B+C+D identifiquen els VERITABLES candidats: voltes amb\n"
        "    tokens hapax (RARE) i/o desviacions de runs respecte H0.\n"
        "  · 'candidats_steganografia.csv' es el mapa per la inspeccio quali-\n"
        "    tativa del facsimil: contrastar margens, marques, canvis de tinta,\n"
        "    dibuixos a les voltes llistades."
    )

    print(f"\n[OK] Escrit: {OUT_JSON.relative_to(BASE)}")
    print(f"[OK] Escrit: {OUT_CSV.relative_to(BASE)}")
    print(f"[OK] Escrit: {OUT_CSV_CAND.relative_to(BASE)}")


if __name__ == "__main__":
    main()
