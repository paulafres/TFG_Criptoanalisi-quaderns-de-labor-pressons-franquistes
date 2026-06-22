# -*- coding: utf-8 -*-
"""
CONFIG CENTRAL DEL PIPELINE
============================

Aquest mòdul concentra:
  · BASE i totes les rutes (entrades i sortides per fase)
  · Constants compartides entre fases (RANDOM_STATE)

Qualsevol script del pipeline ha d'importar les seves rutes d'aquí
en lloc d'hardcoded-les. Si es reanomena una carpeta, només s'edita
aquest fitxer.
"""

from pathlib import Path

# ─── Rutes base ──────────────────────────────────────────────────────────
# config.py viu dins de "SCRIPTS .python/", per tant parents[1] = "PIPELINE/"
BASE = Path(__file__).resolve().parents[1]
PY_DIR = Path(__file__).resolve().parent          # "SCRIPTS .python/"

# =========================================================================
# 4 CORPUS: Quin volem executar?
# =========================================================================
# Opcions vàlides: "CM", "MS", "ND", "P", "TOTS"
CORPUS_ACTIU = "P" 
# =========================================================================

# ─── Enrutament Intel·ligent ─────────────────────────────────────────────
CUADERNOS_DIR = BASE / "CUADERNOS .txt"

if CORPUS_ACTIU == "TOTS":
    # Llegeix de TOTS els subdirectoris alhora
    TEIXITS_DIR = CUADERNOS_DIR
    OUT_ROOT = BASE / "RESULTATS_TOTS"
else:
    # Llegeix només de la carpeta específica (ex: "CM teixits .txt")
    TEIXITS_DIR = CUADERNOS_DIR / f"{CORPUS_ACTIU} teixits .txt"
    OUT_ROOT = BASE / f"RESULTATS_{CORPUS_ACTIU}"


# ─── Carpetes de resultats per fase ──────────────────────────────────────
RES_FASE_1     = OUT_ROOT / "res_fase1"
RES_FASE_2     = OUT_ROOT / "res_fase2"
RES_FASE_3_1   = OUT_ROOT / "res_fase3.1"
RES_FASE_3_1B  = OUT_ROOT / "res_fase3.1b"
RES_FASE_3_2   = OUT_ROOT / "res_fase3.2"
RES_FASE_3_2B  = OUT_ROOT / "res_fase3.2b"
RES_FASE_3_3   = OUT_ROOT / "res_fase3.3"
RES_FASE_3_4   = OUT_ROOT / "res_fase3.4"
RES_FASE_4_1   = OUT_ROOT / "res_fase4.1"
RES_FASE_4_2   = OUT_ROOT / "res_fase4.2"
RES_FASE_4_3   = OUT_ROOT / "res_fase4.3"
RES_FASE_4_4   = OUT_ROOT / "res_fase4.4"
RES_FASE_5_1   = OUT_ROOT / "res_fase5.1"
RES_FASE_5_1B  = OUT_ROOT / "res_fase5.1b"
RES_FASE_5_2   = OUT_ROOT / "res_fase5.2"
RES_FASE_5_2B  = OUT_ROOT / "res_fase5.2b"
RES_FASE_5_3   = OUT_ROOT / "res_fase5.3"
RES_FASE_6     = OUT_ROOT / "res_fase6"

LOG_DIR = OUT_ROOT / "res_logs"


# ─── Fitxers d'entrada/sortida principals (els compartits entre fases) ──
# Fase 1
F1_TEIXITS_DATASET     = RES_FASE_1 / "teixits_dataset.json"
F1_PATRONS_COMPARTITS  = RES_FASE_1 / "taula_patrons_compartits.csv"
F1_SIMBOLS_INDIVIDUALS = RES_FASE_1 / "taula_simbols_individuals.csv"

# Fase 2
F2_INSTRUCTION_DATASET = RES_FASE_2 / "instruction_dataset.json"
F2_TOKEN_INVENTORY     = RES_FASE_2 / "token_inventory.json"

# Fase 3
F3_1_MARKOV_MODEL          = RES_FASE_3_1 / "markov_model.json"
F3_1B_TRANSITION_SIGNIF    = RES_FASE_3_1B / "transition_significance.json"
F3_2_DISTRIB_CLASSES       = RES_FASE_3_2 / "distributional_classes.json"
F3_2B_CLUSTERING_STABILITY = RES_FASE_3_2B / "clustering_stability.json"
F3_3_VOLTA_SIGNATURES      = RES_FASE_3_3 / "volta_signatures.json"
F3_4_VOLTA_CLASSES         = RES_FASE_3_4 / "volta_classes.json"
F3_4_SEQUENCIES_TEIXITS    = RES_FASE_3_4 / "sequencies_teixits.json"

# Fase 4
F4_1_MACRO_GRAMMAR              = RES_FASE_4_1 / "macro_grammar.json"
F4_1_MACRO_SURPRISAL_PER_VOLTA  = RES_FASE_4_1 / "macro_surprisal_per_volta.csv"
F4_2_MICRO_GRAMMAR              = RES_FASE_4_2 / "micro_grammar.json"
F4_3_NULL_MODEL                 = RES_FASE_4_3 / "null_model_permutation.json"
F4_4_LOO_CV_JSON                = RES_FASE_4_4 / "loo_cv_teixits.json"
F4_4_LOO_CV_CSV                 = RES_FASE_4_4 / "loo_cv_teixits.csv"

# Fase 5
F5_1_ANOMALY_REPORT_JSON    = RES_FASE_5_1 / "anomaly_report.json"
F5_1_ANOMALY_REPORT_CSV     = RES_FASE_5_1 / "anomaly_report.csv"
F5_1_ANOMALY_TEIXITS_CSV    = RES_FASE_5_1 / "anomaly_teixits.csv"
F5_1B_CRYPTO_VALIDATION   = RES_FASE_5_1B / "crypto_validation.json"
F5_1B_CANDIDATS_STEG_CSV  = RES_FASE_5_1B / "candidats_steganografia.csv"
F5_2_SEGMENTS_JSON      = RES_FASE_5_2 / "anomaly_segments.json"
F5_2_SEGMENTS_CSV       = RES_FASE_5_2 / "anomaly_segments.csv"
F5_2_REPORT_TXT         = RES_FASE_5_2 / "anomaly_segments_report.txt"
# Fase 5.2b (triangulació 5.1 gramaticals × 5.1b lèxiques sobre segments 5.2)
F5_2B_TRIANGULACIO_JSON   = RES_FASE_5_2B / "triangulacio_segments.json"
F5_2B_TRIANGULACIO_CSV    = RES_FASE_5_2B / "triangulacio_segments.csv"
F5_2B_TRIANGULACIO_REPORT = RES_FASE_5_2B / "triangulacio_segments_report.txt"
F5_2B_ORFENES_CSV         = RES_FASE_5_2B / "voltes_orfenes.csv"
F5_3_HIPOTESIS_REFS       = RES_FASE_5_3 / "hipotesis_referencies.json"

# ─── Constants compartides ───────────────────────────────────────────────
RANDOM_STATE = 42


# ─── Helper ──────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> Path:
    """Crea el directori si no existeix i el retorna."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
