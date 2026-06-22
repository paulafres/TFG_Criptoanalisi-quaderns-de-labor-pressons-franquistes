# -*- coding: utf-8 -*-
"""
RESUM GLOBAL DEL PIPELINE (ORDENAT PER SCRIPTS)
===============================================
Llegeix els JSONs de totes les fases exactament en l'ordre d'execució
dels scripts .py i genera un resum a consola i a 'resum_global.txt'.
"""

import json
from pathlib import Path

# Importem les rutes exactes definides al teu config.py
from config import (
    BASE,
    F1_TEIXITS_DATASET, F2_INSTRUCTION_DATASET,
    F3_1_MARKOV_MODEL, F3_1B_TRANSITION_SIGNIF,
    F3_2_DISTRIB_CLASSES, F3_2B_CLUSTERING_STABILITY,
    F3_3_VOLTA_SIGNATURES, F3_4_VOLTA_CLASSES,
    F4_1_MACRO_GRAMMAR, F4_2_MICRO_GRAMMAR,
    F4_3_NULL_MODEL, F4_4_LOO_CV_JSON,
    F5_1_ANOMALY_REPORT_JSON, F5_1B_CRYPTO_VALIDATION,
    F5_2_SEGMENTS_JSON, F5_2B_TRIANGULACIO_JSON,
    F5_3_HIPOTESIS_REFS
)

OUT_TXT = BASE / "resum_global.txt"

def carrega_json(ruta: Path) -> dict:
    if ruta and ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        # Netegem el fitxer si existia
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write("")

    def log(self, text=""):
        print(text)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def separa(self, titol: str):
        self.log("\n" + "=" * 75)
        self.log(f" {titol}")
        self.log("=" * 75)

def main():
    logger = Logger(OUT_TXT)
    logger.log("\n" + "★" * 75)
    logger.log(" RESUM GLOBAL DE RESULTATS DEL TFG")
    logger.log("★" * 75)

    # ─── FASE 1: Corpus Normalizer ─────────────────────────────────────────
    logger.separa("fase1_corpus_normalizer.py")
    f1 = carrega_json(F1_TEIXITS_DATASET)
    if f1:
        n_teixits = len(f1)
        n_voltes = sum(t.get("num_instruccions", 0) for t in f1)
        logger.log(f"  Teixits processats      : {n_teixits}")
        logger.log(f"  Instruccions (voltes)   : {n_voltes}")
    else:
        logger.log("  [!] Dades no trobades.")

    # ─── FASE 2: Token Inventory ───────────────────────────────────────────
    logger.separa("fase2_token_inventory.py")
    f2 = carrega_json(F2_INSTRUCTION_DATASET)
    if f2:
        logger.log(f"  Instruccions expandides : {len(f2)} voltes avaluables")
    else:
        logger.log("  [!] Dades no trobades.")

    # ─── FASE 3.1: Markov Model ────────────────────────────────────────────
    logger.separa("fase3.1_markov_model.py")
    f31 = carrega_json(F3_1_MARKOV_MODEL)
    if f31:
        md = f31.get("metadades", {})
        logger.log(f"  Volum total tokens      : {md.get('num_tokens_corpus')}")
        logger.log(f"  Vocabulari (tokens únics): {md.get('num_tokens_unics')}")
        logger.log(f"  Bigrames únics observats: {md.get('num_bigrames_unics')}")

    # ─── FASE 3.1b: Transition Significance ────────────────────────────────
    logger.separa("fase3.1b_transition_significance.py")
    f31b = carrega_json(F3_1B_TRANSITION_SIGNIF)
    if f31b:
        md = f31b.get("metadades", {})
        logger.log(f"  Transicions testades    : {md.get('num_bigrames_testats')}")
        logger.log(f"  Significatives (FDR)    : {md.get('num_significatius_FDR')} (no atzaroses)")

    # ─── FASE 3.2: Distributional Classes ──────────────────────────────────
    logger.separa("fase3.2_distributional_classes.py")
    f32 = carrega_json(F3_2_DISTRIB_CLASSES)
    if f32:
        md = f32.get("metadades", {})
        logger.log(f"  Classes de tokens (K)   : {md.get('k_optim')} classes funcionals (C0-C7)")
        logger.log(f"  Silhouette Òptim        : {md.get('silhouette_optim')}")
        logger.log(f"  Tokens aïllats (C_RARE) : {md.get('num_tokens_rare')} hàpax")

    # ─── FASE 3.2b: Clustering Stability ───────────────────────────────────
    logger.separa("fase3.2b_clustering_stability.py")
    f32b = carrega_json(F3_2B_CLUSTERING_STABILITY)
    if f32b:
        res = f32b.get("resultats", {})
        logger.log(f"  Estabilitat ARI (mitja) : {res.get('ARI_mitja')} ± {res.get('ARI_sd')}")
        logger.log(f"  Interval Conf. (IC95%)  : {res.get('ARI_IC95')}")

    # ─── FASE 3.3: Volta Signatures ────────────────────────────────────────
    logger.separa("fase3.3_volta_signatures.py")
    f33 = carrega_json(F3_3_VOLTA_SIGNATURES)
    if f33:
        sg = f33.get("metadades", {}).get("salts_globals", {})
        logger.log(f"  Salts globals avaluats  : {sg.get('n')} transicions")
        logger.log(f"  Distància cosinus mitja : {sg.get('mitjana')} ± {sg.get('desv_std')}")

    # ─── FASE 3.4: Volta Clustering ────────────────────────────────────────
    logger.separa("fase3.4_volta_clustering.py")
    f34 = carrega_json(F3_4_VOLTA_CLASSES)
    if f34:
        md = f34.get("metadades", {})
        logger.log(f"  Tipologies volta (VTx)  : {md.get('k_optim')} tipus estructurals (V0-V4)")
        logger.log(f"  Silhouette Òptim        : {md.get('silhouette_optim')}")

    # ─── FASE 4.1: Macro Grammar ───────────────────────────────────────────
    logger.separa("fase4.1_macro_grammar.py")
    f41 = carrega_json(F4_1_MACRO_GRAMMAR)
    if f41:
        md = f41.get("metadades", {})
        logger.log(f"  Contextos macro vistos  : {md.get('n_contexts')}")
        logger.log(f"  Entropia MACRO (H_global): {md.get('H_global_bits')} bits")

    # ─── FASE 4.2: Micro Grammar ───────────────────────────────────────────
    logger.separa("fase4.2_micro_grammar.py")
    f42 = carrega_json(F4_2_MICRO_GRAMMAR)
    if f42:
        md = f42.get("metadades", {})
        logger.log(f"  Contextos micro vistos  : {md.get('n_contexts')}")
        logger.log(f"  Entropia MICRO (H_global): {md.get('H_global_bits')} bits")

    # ─── FASE 4.3: Null Model ──────────────────────────────────────────────
    logger.separa("fase4.3_null_model.py")
    f43 = carrega_json(F4_3_NULL_MODEL)
    if f43:
        res = f43.get("resultats", {})
        logger.log(f"  H_observat vs H_null    : {res.get('H_observat')} vs {res.get('H_null_mitja')} bits")
        logger.log(f"  Z-score (Permutacions)  : {res.get('z_score')}")
        logger.log(f"  P-valor empíric         : {res.get('p_value_empirica')}")

    # ─── FASE 4.4: LOO-CV ──────────────────────────────────────────────────
    logger.separa("fase4.4_loo_cv.py")
    f44 = carrega_json(F4_4_LOO_CV_JSON)
    if f44:
        md = f44.get("metadades", {})
        logger.log(f"  Surprisal LOO (mitjana) : {md.get('surprisal_loo_mitjana')} ± {md.get('surprisal_loo_sd')} bits/token")

    # ─── FASE 5.1: Anomaly Report ──────────────────────────────────────────
    logger.separa("fase5.1_anomaly_report.py")
    f51 = carrega_json(F5_1_ANOMALY_REPORT_JSON)
    if f51:
        md = f51.get("metadades", {})
        pc1 = md.get("pca", {}).get("variancia_pc1", 0)
        cats = [a["categoria"] for a in f51.get("anomalies", [])]
        logger.log(f"  Variància recollida PC1 : {pc1 * 100:.2f}%")
        logger.log(f"  Voltes CRÍTICA (>=p99)  : {cats.count('CRÍTICA')}")
        logger.log(f"  Voltes ALTA (>=p97)     : {cats.count('ALTA')}")

    # ─── FASE 5.1b: Crypto Validation (Voltes individuals) ─────────────────
    logger.separa("fase5.1b_crypto_validation.py")
    f51b = carrega_json(F5_1B_CRYPTO_VALIDATION)
    if f51b:
        sa = f51b.get("seccio_A_classic_sobre_anomalies_fase5", {}).get("veredicte", {})
        logger.log(f"  IC Anomalies vs Corpus  : {sa.get('ratio')} (Ràtio)")
        logger.log(f"  Diagnòstic Voltes       : {sa.get('diagnostic')}")

    # ─── FASE 5.2: Segment Anomaly Detection ───────────────────────────────
    logger.separa("fase5.2_segment_anomaly_detection.py")
    f52 = carrega_json(F5_2_SEGMENTS_JSON)
    if f52:
        md = f52.get("metadades", {})
        logger.log(f"  Segments contigus cand. : {md.get('n_segments_candidats')}")
        logger.log(f"  Significatius (FDR <=α) : {md.get('n_segments_significatius')}")

    # ─── FASE 5.2b: Triangulació gramatical × lèxica ───────────────────────
    logger.separa("fase5.2b_crypto_segments.py")
    f52b = carrega_json(F5_2B_TRIANGULACIO_JSON)
    if f52b:
        md = f52b.get("metadades", {})
        logger.log(f"  Segments triangulats    : {md.get('n_segments')}")
        logger.log(f"  Convergència (5.1∩5.1b) : {md.get('n_convergencia')}")
        logger.log(f"  Perfil lèxic (abecedari): {md.get('n_perfil_lexic')}")
        logger.log(f"  Perfil gramatical       : {md.get('n_perfil_gramatical')}")
        logger.log(f"  Mixt / sense senyal     : {md.get('n_mixt')} / {md.get('n_sense_senyal')}")
        orf = md.get("orfenes_per_classe", {})
        logger.log(f"  Voltes òrfenes (fora seg): {md.get('n_voltes_atipiques_orfenes')} "
                   f"(conv={orf.get('CONVERGENT', 0)}, lex={orf.get('LEXICA', 0)}, "
                   f"gram={orf.get('GRAMATICAL', 0)})")

    # ─── FASE 5.3: Hipotesis Referencies ───────────────────────────────────
    logger.separa("fase5.3_hipotesis_referencies.py")
    f53 = carrega_json(F5_3_HIPOTESIS_REFS)
    if f53:
        logger.log("  Hipòtesi A (Morse Tèxtil)   : REBUTJADA")
        logger.log("  Hipòtesi B (Anom. Belgues)  : REBUTJADA")
        
        n_idem = len([c for c in f53.get("manolita", {}).get("coincidencies_v71", []) if c["tipus"] == "IDÈNTICA"])
        estat_c = "CONFIRMADA" if n_idem > 0 else "PENDENT"
        logger.log(f"  Hipòtesi C (Manolita/PCE)   : {estat_c} ({n_idem} coincidència caràcter per caràcter)")

    logger.log("\n" + "─" * 75)
    logger.log(f" [OK] Resum completat! El tens guardat a: {OUT_TXT.name}")
    logger.log("─" * 75 + "\n")

if __name__ == "__main__":
    main()